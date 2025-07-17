#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>

#include "ATen/ops/empty_like.h"
#include "c10/core/ScalarType.h"
#include "c10/util/BFloat16.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <ATen/Dispatch.h>
#include <torch/all.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>
#include <torch_musa/csrc/core/MUSAGuard.h>
#include <cstdint>

#include "common/utils.muh"

static constexpr int iobits = 512;

namespace musa_extension {
namespace {

// same as PyTorch's modes
enum class ReductionMode { None, Mean, Sum };

template <typename T>
struct CppTypeTraits;

template <>
struct CppTypeTraits<double> {
  using type = double;
};

template <>
struct CppTypeTraits<float> {
  using type = float;
};

template <>
struct CppTypeTraits<at::Half> {
  using type = __half;
};

template <>
struct CppTypeTraits<at::BFloat16> {
  using type = __mt_bfloat16;
};

// the online_softmax_kernel and cross_entropy_loss_kernel should be less
// efficient when number of tokens and vocab_size is small, but considering that
// the scenrio we are oriented towards is LLMs training, the current
// implementation is efficient enough.

// Online normalizer calculation for softmax: https://arxiv.org/pdf/1805.02867
template <typename scalar_t, typename index_t = int64_t, bool Aligned>
__global__ void online_softmax_kernel(
    const scalar_t* logits,
    const index_t* targets,
    scalar_t* output,
    int64_t num_tokens,
    int64_t vocab_size,
    int vocab_start_idx,
    int vocab_end_idx) {
  uint32_t bid = blockIdx.x * blockDim.y + threadIdx.y;
  uint32_t tid = threadIdx.x;

  using AccessType = typename VecTraits<scalar_t, iobits>::AccessType;
  constexpr int vlen = VecTraits<scalar_t, iobits>::vlen;
  constexpr int ELEMENTS_PER_WARP = vlen * WARP_SIZE;

  if (bid >= num_tokens) {
    return;
  }

  const scalar_t* logits_base_load_ptr = logits + bid * vocab_size;
  const index_t* targets_base_load_ptr = targets + bid;
  scalar_t* output_base_store_ptr = output + bid * 3;
  float maxval = -INFINITY;
  float sumval = 0.0f;

#define ONLINE_CALCULATION(cur_val)                       \
  if (cur_val > maxval) {                                 \
    float pre_maxval = maxval;                            \
    maxval = cur_val;                                     \
    sumval = sumval * __expf(pre_maxval - maxval) + 1.0f; \
  } else {                                                \
    sumval += __expf(cur_val - maxval);                   \
  }

  // each warp process one row
  for (int offset = tid * vlen; offset < vocab_size;
       offset += ELEMENTS_PER_WARP) {
    if constexpr (Aligned) {
      const AccessType regs =
          *reinterpret_cast<const AccessType*>(logits_base_load_ptr + offset);
#pragma unroll
      for (int i = 0; i < vlen; i++) {
        float cur_val = static_cast<float>(regs.data[i]);
        ONLINE_CALCULATION(cur_val);
      }

    } else {
      if (offset + vlen <= vocab_size) {
        const AccessType regs =
            *reinterpret_cast<const AccessType*>(logits_base_load_ptr + offset);
#pragma unroll
        for (int i = 0; i < vlen; i++) {
          float cur_val = static_cast<float>(regs.data[i]);
          ONLINE_CALCULATION(cur_val);
        }
      } else {
        int idx = offset;
        while (idx < vocab_size) {
          float cur_val = static_cast<float>(logits_base_load_ptr[idx]);
          ONLINE_CALCULATION(cur_val);
          idx++;
        }
      }
    }
  }

  // do warp reduce
  float other_sum;
  float other_max;
#pragma unroll
  for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
    other_max = musa_shfl_xor_sync<float, WARP_SIZE>(maxval, offset);
    other_sum = musa_shfl_xor_sync<float, WARP_SIZE>(sumval, offset);

    if (!(isinf(maxval) && isinf(other_max))) {
      if (other_max > maxval) {
        sumval *= __expf(maxval - other_max);
        maxval = other_max;
      } else {
        other_sum *= __expf(other_max - maxval);
      }
      sumval += other_sum;
    }
  }

  if (threadIdx.x == 0) {
    const index_t target_y = targets_base_load_ptr[0];
    float x_y = -INFINITY;
    if (target_y >= vocab_start_idx && target_y < vocab_end_idx) {
      x_y =
          static_cast<float>(logits_base_load_ptr[target_y - vocab_start_idx]);
    }

    output_base_store_ptr[0] = maxval;
    output_base_store_ptr[1] = sumval;
    output_base_store_ptr[2] = x_y;
  }
}

// scalar_t should be float ?
template <
    typename scalar_t,
    typename index_t,
    bool Aligned,
    ReductionMode REDUCTION_TYPE>
__global__ void cross_entropy_loss_kernel(
    scalar_t* logits,
    const index_t* targets,
    const scalar_t* gathered_max_sum_y,
    scalar_t* losses,
    int64_t num_tokens,
    int64_t vocab_size,
    int vocab_start_idx,
    int vocab_end_idx,
    int world_size) {
  uint32_t bid = blockIdx.x * blockDim.y + threadIdx.y;
  uint32_t tid = threadIdx.x;

  using AccessType = typename VecTraits<scalar_t, iobits>::AccessType;
  constexpr int vlen = VecTraits<scalar_t, iobits>::vlen;
  constexpr int ELEMENTS_PER_WARP = vlen * WARP_SIZE;

  if (bid >= num_tokens) {
    return;
  }

  // will write gradients to logits, so the underlying data of
  // logits_base_load_ptr should be writeable
  scalar_t* logits_base_load_ptr = logits + bid * vocab_size;
  const scalar_t* gathered_max_sum_y_base_load_ptr =
      gathered_max_sum_y + bid * world_size * 3;
  scalar_t* losses_base_store_ptr = losses + bid;
  const index_t* targets_base_load_ptr = targets + bid;

  // assert(world_size <= WARP_SIZE)
  const bool valid_tid = tid < world_size;
  float maxval =
      valid_tid ? gathered_max_sum_y_base_load_ptr[tid * 3] : -INFINITY;
  float expsumval =
      valid_tid ? gathered_max_sum_y_base_load_ptr[tid * 3 + 1] : 0.0f;
  float logits_y =
      valid_tid ? gathered_max_sum_y_base_load_ptr[tid * 3 + 2] : -INFINITY;

  float other_max;
  float other_expsum;
  float other_logits_y;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
    other_max = musa_shfl_xor_sync<float, WARP_SIZE>(maxval, offset);
    other_expsum = musa_shfl_xor_sync<float, WARP_SIZE>(expsumval, offset);
    other_logits_y = musa_shfl_xor_sync<float, WARP_SIZE>(logits_y, offset);

    if (!(isinf(maxval) && isinf(other_max))) {
      if (other_max > maxval) {
        expsumval *= __expf(maxval - other_max);
        maxval = other_max;
      } else {
        other_expsum *= __expf(other_max - maxval);
      }
      expsumval += other_expsum;
    }
    logits_y = fmaxf(logits_y, other_logits_y);
  }

  // now each thread gets the global maxval, expsum and valid logits_y,
  // will calculate gradients and write back to logits' storage
  // TODO: implement case with lable_smoothing > 0 and ignore_idx
  // float coefficient = 1.0f / expsumval / num_tokens;  // mean reduction case
  float coefficient = 1.0f / expsumval;
  if constexpr (REDUCTION_TYPE == ReductionMode::Mean) {
    coefficient = coefficient / num_tokens;
  }
  for (int offset = tid * vlen; offset < vocab_size;
       offset += ELEMENTS_PER_WARP) {
    if constexpr (Aligned) {
      AccessType* logits_cur_vec_ptr =
          reinterpret_cast<AccessType*>(logits_base_load_ptr + offset);
      AccessType logits_regs = *logits_cur_vec_ptr;
      AccessType gradients_regs;
#pragma unroll
      for (int i = 0; i < vlen; i++) {
        gradients_regs.data[i] =
            __expf(logits_regs.data[i] - maxval) * coefficient;
      }
      // *reinterpret_cast<AccessType*>(logits_base_load_ptr + offset) =
      // gradients_regs;
      logits_cur_vec_ptr[0] = gradients_regs;
    } else {
      if (offset + vlen <= vocab_size) {
        AccessType* logits_cur_vec_ptr =
            reinterpret_cast<AccessType*>(logits_base_load_ptr + offset);
        AccessType logits_regs = *logits_cur_vec_ptr;
        AccessType gradients_regs;
#pragma unroll
        for (int i = 0; i < vlen; i++) {
          gradients_regs.data[i] =
              __expf(logits_regs.data[i] - maxval) * coefficient;
        }
        logits_cur_vec_ptr[0] = gradients_regs;
      } else {
        int idx = offset;
        while (idx < vocab_size) {
          scalar_t* logits_cur_ptr = logits_base_load_ptr + idx;
          scalar_t gradient = __expf((*logits_cur_ptr) - maxval) * coefficient;
          logits_cur_ptr[0] = gradient;
          idx++;
        }
      }
    }
  }

  if (threadIdx.x == 0) {
    float loss = -logits_y + maxval + __logf(expsumval);
    losses_base_store_ptr[0] = static_cast<scalar_t>(loss);

    const index_t target_y = targets_base_load_ptr[0];
    if (target_y >= vocab_start_idx && target_y < vocab_end_idx) {
      float x_y =
          static_cast<float>(logits_base_load_ptr[target_y - vocab_start_idx]);
      // x_y += -(1.0f - label_smoothing) / num_tokens;
      if constexpr (REDUCTION_TYPE == ReductionMode::Mean) {
        x_y += (-1.0f / num_tokens);
      } else if constexpr (REDUCTION_TYPE == ReductionMode::Sum) {
        x_y += -1.0f;
      }
      logits_base_load_ptr[target_y - vocab_start_idx] =
          static_cast<scalar_t>(x_y);
    }
  }
}

// apply global reduction over losses
template <
    typename T,
    typename ACC_TYPE,
    ReductionMode REDUCTION_TYPE,
    bool Aligned,
    int BLOCK_SIZE = 512>
__global__ void normalize_losses_kernel(
    const T* losses_ptr,
    T* output_ptr,
    int numel) {
  uint32_t tid = threadIdx.x;
  uint32_t warp_id = tid / WARP_SIZE;
  uint32_t lane_id = tid % WARP_SIZE;

  using AccessType = typename VecTraits<T, iobits>::AccessType;
  constexpr int vlen = VecTraits<T, iobits>::vlen;
  constexpr int num_warps = BLOCK_SIZE / WARP_SIZE;
  constexpr int ELEMENTS_PER_BLOCK = vlen * BLOCK_SIZE;

  __shared__ ACC_TYPE shm[WARP_SIZE];

  ACC_TYPE sum = static_cast<ACC_TYPE>(0.0f);

  for (int offset = tid * vlen; offset < numel; offset += ELEMENTS_PER_BLOCK) {
    if constexpr (Aligned) {
      const AccessType regs =
          *reinterpret_cast<const AccessType*>(losses_ptr + offset);
#pragma unroll
      for (int i = 0; i < vlen; i++) {
        sum += static_cast<ACC_TYPE>(regs.data[i]);
      }
    } else {
      if (offset + vlen <= numel) {
        const AccessType regs =
            *reinterpret_cast<const AccessType*>(losses_ptr + offset);
#pragma unroll
        for (int i = 0; i < vlen; i++) {
          sum += static_cast<ACC_TYPE>(regs.data[i]);
        }
      } else {
        int idx = offset;
        while (idx < numel) {
          sum += losses_ptr[idx++];
        }
      }
    }
  }

  sum = warpReduce<ACC_TYPE, Add, WARP_SIZE>(sum);
  if (lane_id == 0) {
    shm[warp_id] = sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    sum = lane_id < num_warps ? shm[lane_id] : static_cast<ACC_TYPE>(0.0f);
    for (int mask = 1; mask < next_power_of_two(num_warps); mask *= 2) {
      ACC_TYPE other_sum = musa_shfl_xor_sync<ACC_TYPE, WARP_SIZE>(sum, mask);
      sum += other_sum;
    }

    if (tid == 0) {
      if constexpr (REDUCTION_TYPE == ReductionMode::Sum) {
        output_ptr[0] = static_cast<T>(sum);
      } else if constexpr (REDUCTION_TYPE == ReductionMode::Mean) {
        output_ptr[0] = static_cast<T>(sum / (float)numel);
      }
    }
  }
}

} // anonymous namespace
} // namespace musa_extension

at::Tensor online_softmax(at::Tensor logits, at::Tensor targets, int rank) {
  TORCH_CHECK(
      logits.dim() == 2 && targets.dim() == 1,
      "Expect logits have 2D shape and targets have 1D shape");

  TORCH_CHECK(logits.is_contiguous(), "only support contiguous logits");
  TORCH_CHECK(targets.is_contiguous(), "only support contiguous targets");

  const int64_t num_tokens = logits.size(0);
  const int64_t vocab_size = logits.size(1);

  // create output, num_tokens * 3
  const auto output_options =
      logits.options().memory_format(c10::MemoryFormat::Contiguous);
  at::Tensor output = at::empty({num_tokens, 3}, output_options);

  const int64_t vocab_start_idx = rank * vocab_size;
  const int64_t vocab_end_idx = (rank + 1) * vocab_size;

  // AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16,
  // logits.scalar_type(), "online_softmax", [&] {
  AT_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "online_softmax", [&] {
    AT_DISPATCH_INDEX_TYPES(targets.scalar_type(), "online_softmax_index", [&] {
      using cpp_t = musa_extension::CppTypeTraits<scalar_t>::type;

      // each warp process one row
      const uint32_t threads_per_block = 256;
      const uint32_t warp_size = at::musa::warp_size();
      const uint32_t warps_per_block = threads_per_block / warp_size;
      dim3 block = {warp_size, warps_per_block, 1};
      dim3 grid = {
          ((uint32_t)num_tokens + warps_per_block - 1) / warps_per_block, 1, 1};
      const auto stream = at::musa::getCurrentMUSAStream();
      bool aligned =
          (vocab_size % musa_extension::VecTraits<cpp_t, iobits>::vlen) == 0;

      if (aligned) {
        musa_extension::online_softmax_kernel<cpp_t, index_t, true>
            <<<grid, block, 0, stream>>>(
                static_cast<const cpp_t*>(logits.data_ptr()),
                static_cast<const index_t*>(targets.data_ptr()),
                static_cast<cpp_t*>(output.data_ptr()),
                num_tokens,
                vocab_size,
                vocab_start_idx,
                vocab_end_idx);
      } else {
        musa_extension::online_softmax_kernel<cpp_t, index_t, false>
            <<<grid, block, 0, stream>>>(
                static_cast<const cpp_t*>(logits.data_ptr()),
                static_cast<const index_t*>(targets.data_ptr()),
                static_cast<cpp_t*>(output.data_ptr()),
                num_tokens,
                vocab_size,
                vocab_start_idx,
                vocab_end_idx);
      }
      C10_MUSA_KERNEL_LAUNCH_CHECK();
    });
  });

  return output;
}

at::Tensor cross_entropy_loss(
    at::Tensor logits,
    at::Tensor targets,
    at::Tensor gathered_max_sum_y,
    int rank,
    int world_size,
    const std::string& reduction = "mean") {
  const int64_t num_tokens = logits.size(0);
  const int64_t vocab_size = logits.size(1);

  // create output of loss, num_tokens * 1
  const auto loss1d_options =
      logits.options().memory_format(c10::MemoryFormat::Contiguous);
  at::Tensor loss1d = at::empty({num_tokens}, loss1d_options);
  at::Tensor output = loss1d;

  const int64_t vocab_start_idx = rank * vocab_size;
  const int64_t vocab_end_idx = (rank + 1) * vocab_size;

  AT_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "cross_entropy_loss", [&] {
    AT_DISPATCH_INDEX_TYPES(
        targets.scalar_type(), "cross_entropy_loss_index", [&] {
          using cpp_t = musa_extension::CppTypeTraits<scalar_t>::type;

          // each warp process one row, each block may have multiple warps
          const uint32_t threads_per_block = 256;
          const uint32_t warp_size = at::musa::warp_size();
          const uint32_t warps_per_block = threads_per_block / warp_size;
          dim3 block = {warp_size, warps_per_block, 1};
          dim3 grid = {
              ((uint32_t)num_tokens + warps_per_block - 1) / warps_per_block,
              1,
              1};
          const auto stream = at::musa::getCurrentMUSAStream();
          bool aligned = (vocab_size %
                          musa_extension::VecTraits<cpp_t, iobits>::vlen) == 0;

#define LAUNCH_CROSS_ENTROPY_LOSS_KERNEL(r_mode)                             \
  if (aligned) {                                                             \
    musa_extension::cross_entropy_loss_kernel<cpp_t, index_t, true, r_mode>  \
        <<<grid, block, 0, stream>>>(                                        \
            static_cast<cpp_t*>(logits.data_ptr()),                          \
            static_cast<const index_t*>(targets.data_ptr()),                 \
            static_cast<const cpp_t*>(gathered_max_sum_y.data_ptr()),        \
            static_cast<cpp_t*>(loss1d.data_ptr()),                          \
            num_tokens,                                                      \
            vocab_size,                                                      \
            vocab_start_idx,                                                 \
            vocab_end_idx,                                                   \
            world_size);                                                     \
  } else {                                                                   \
    musa_extension::cross_entropy_loss_kernel<cpp_t, index_t, false, r_mode> \
        <<<grid, block, 0, stream>>>(                                        \
            static_cast<cpp_t*>(logits.data_ptr()),                          \
            static_cast<const index_t*>(targets.data_ptr()),                 \
            static_cast<const cpp_t*>(gathered_max_sum_y.data_ptr()),        \
            static_cast<cpp_t*>(loss1d.data_ptr()),                          \
            num_tokens,                                                      \
            vocab_size,                                                      \
            vocab_start_idx,                                                 \
            vocab_end_idx,                                                   \
            world_size);                                                     \
  }                                                                          \
  C10_MUSA_KERNEL_LAUNCH_CHECK();

      // (maybe) Do the final reduction on losses

#define LAUNCH_NORMALIZE_LOSS_KERNEL(r_mode)                                \
  {                                                                         \
    output = at::empty({1}, loss1d_options);                                \
    constexpr uint32_t threads_per_block = 1024;                            \
    dim3 block = {threads_per_block, 1, 1};                                 \
    dim3 grid = {1, 1, 1};                                                  \
    bool aligned =                                                          \
        (num_tokens % musa_extension::VecTraits<cpp_t, iobits>::vlen) == 0; \
    if (aligned) {                                                          \
      musa_extension::normalize_losses_kernel<                              \
          cpp_t,                                                            \
          float,                                                            \
          r_mode,                                                           \
          true,                                                             \
          threads_per_block><<<grid, block, 0, stream>>>(                   \
          static_cast<const cpp_t*>(loss1d.data_ptr()),                     \
          static_cast<cpp_t*>(output.data_ptr()),                           \
          num_tokens);                                                      \
    } else {                                                                \
      musa_extension::normalize_losses_kernel<                              \
          cpp_t,                                                            \
          float,                                                            \
          r_mode,                                                           \
          false,                                                            \
          threads_per_block><<<grid, block, 0, stream>>>(                   \
          static_cast<const cpp_t*>(loss1d.data_ptr()),                     \
          static_cast<cpp_t*>(output.data_ptr()),                           \
          num_tokens);                                                      \
    }                                                                       \
    C10_MUSA_KERNEL_LAUNCH_CHECK();                                         \
  }

          if (reduction == "mean") {
            LAUNCH_CROSS_ENTROPY_LOSS_KERNEL(
                musa_extension::ReductionMode::Mean);
            LAUNCH_NORMALIZE_LOSS_KERNEL(musa_extension::ReductionMode::Mean);
          } else if (reduction == "sum") {
            LAUNCH_CROSS_ENTROPY_LOSS_KERNEL(
                musa_extension::ReductionMode::Sum);
            LAUNCH_NORMALIZE_LOSS_KERNEL(musa_extension::ReductionMode::Sum);
          } else {
            TORCH_CHECK(false, "Unsupported reduction mode: ", reduction);
          }
        });
  });

  return output;
}
