#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/musa/cub.h>

#include <ATen/musa/ThrustAllocator.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/ops/musa/EmbeddingBackwardKernel.muh"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace native {

using at::musa::FastDivmod;
using at::musa::VecType;

namespace {
constexpr int NROWS_PER_THREAD = 10;

__host__ __device__ __forceinline__ int64_t ceil_div(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <typename index_t>
int64_t EmbeddingBackwardMUSAKernelUniqueByKey(
    const Tensor& sorted_indices,
    Tensor& segment_offsets) {
  auto stream = at::musa::getCurrentMUSAStream();
  at::musa::ThrustAllocator allocator;
  auto policy = thrust::musa::par(allocator).on(stream);
  const size_t numel = sorted_indices.numel();
  auto sorted_indices_dev =
      thrust::device_ptr<index_t>(sorted_indices.data_ptr<index_t>());
  Tensor dummy =
      at::empty_like(sorted_indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto dummy_dev = thrust::device_ptr<index_t>(dummy.data_ptr<index_t>());
  auto ends = thrust::unique_by_key_copy(
      policy,
      sorted_indices_dev,
      sorted_indices_dev + numel,
      thrust::make_counting_iterator(0),
      dummy_dev,
      thrust::device_ptr<index_t>(segment_offsets.data_ptr<index_t>()));
  return thrust::get<0>(ends) - dummy_dev;
}

template <typename index_t>
__global__ void PartialsPerSegment(
    index_t* partials_per_segment,
    const index_t* segment_offsets,
    int32_t num_of_segments,
    const int64_t numel) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int gsz = blockDim.x * gridDim.x;
  for (int i = gid; i < num_of_segments; i += gsz) {
    const int64_t idx_start = segment_offsets[i];
    const int64_t idx_end =
        (i == num_of_segments - 1) ? numel : segment_offsets[i + 1];
    const int64_t size = idx_end - idx_start;
    partials_per_segment[i] = ceil_div(size, NROWS_PER_THREAD);
  }
}

template <typename index_t>
__global__ void PartialsSegmentOffset(
    int64_t* num_of_partial_segments,
    index_t* partial_segment_offset,
    const index_t* partials_per_segment,
    const index_t* partials_per_segment_offset,
    const index_t* segment_offsets,
    const int32_t num_of_segments) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int gsz = blockDim.x * gridDim.x;

  if (0 == gid) {
    // reduce one kernel launch overhead
    *num_of_partial_segments = partials_per_segment[num_of_segments - 1] +
        partials_per_segment_offset[num_of_segments - 1];
  }

  for (int i = gid; i < num_of_segments; i += gsz) {
    index_t idx = partials_per_segment_offset[i];
    const index_t num_partials = partials_per_segment[i];
    const index_t segment_offset = segment_offsets[i];
    for (int64_t j = 0; j < num_partials; j++) {
      partial_segment_offset[idx++] = segment_offset + j * NROWS_PER_THREAD;
    }
  }
}

template <typename scalar_t, typename index_t, const int64_t vlen>
__global__ void ComputeDwSegmentVector(
    acc_type<scalar_t, true>* dw_seg,
    const scalar_t* dy,
    const index_t* origin_idx,
    const index_t* partials_segment_offset,
    const int num_of_partial_segments,
    const int numel,
    const int64_t stride,
    FastDivmod stride_warped_fastdv) {
  using accscalar_t = acc_type<scalar_t, true>;
  using vec_dtype = VecType<scalar_t, vlen * sizeof(scalar_t) * 8>;
  using vec_acc_dtype = VecType<accscalar_t, vlen * sizeof(accscalar_t) * 8>;

  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t id, feature_offset; // quotient, remainder
  stride_warped_fastdv(id, feature_offset, gid);
  uint32_t feature_offset_vlen = feature_offset * vlen;
  if (feature_offset_vlen >= stride) {
    return;
  }
  if (id >= num_of_partial_segments) {
    return;
  }

  const int idx_start = partials_segment_offset[id];
  const int idx_end = (id == num_of_partial_segments - 1)
      ? numel
      : partials_segment_offset[id + 1];
  if (feature_offset_vlen + vlen <= stride) {
    vec_acc_dtype weight;
    for (int i = idx_start; i < idx_end; i++) {
      const index_t origin_id = origin_idx[i];
      vec_dtype reg_dy =
          vec_dtype::load(dy, origin_id * stride + feature_offset_vlen);
#pragma unroll
      for (int k = 0; k < vlen; k++) {
        weight.val_.elem[k] += (accscalar_t)reg_dy.val_.elem[k];
      }
    }
    vec_acc_dtype::store(dw_seg, id * stride + feature_offset_vlen, weight);
  } else {
    while (feature_offset_vlen < stride) {
      accscalar_t weight = 0;
      for (int i = idx_start; i < idx_end; i++) {
        const index_t origin_id = origin_idx[i];
        weight += dy[origin_id * stride + feature_offset_vlen];
      }
      dw_seg[id * stride + feature_offset_vlen] = weight;
      feature_offset_vlen++;
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void ComputeDwSegment(
    acc_type<scalar_t, true>* dw_seg,
    const scalar_t* dy,
    const index_t* origin_idx,
    const index_t* partials_segment_offset,
    const int num_of_partial_segments,
    const int numel,
    const int64_t stride,
    FastDivmod stride_warped_fastdv) {
  using accscalar_t = acc_type<scalar_t, true>;
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t id, feature_offset; // quotient, remainder
  stride_warped_fastdv(id, feature_offset, gid);
  if (feature_offset >= stride) {
    return;
  }
  if (id >= num_of_partial_segments) {
    return;
  }

  const int idx_start = partials_segment_offset[id];
  const int idx_end = (id == num_of_partial_segments - 1)
      ? numel
      : partials_segment_offset[id + 1];
  accscalar_t weight = 0;
  for (int i = idx_start; i < idx_end; i++) {
    const index_t origin_id = origin_idx[i];
    weight += dy[origin_id * stride + feature_offset];
  }
  dw_seg[id * stride + feature_offset] = weight;
}

template <typename scalar_t, typename index_t, const int64_t vlen>
__global__ void SumAndScatterVector(
    scalar_t* dw,
    const acc_type<scalar_t, true>* dw_segments,
    const index_t* idx,
    const index_t* segment_offsets,
    const index_t* partials_per_segment_offset,
    const int num_of_segments,
    const int num_of_partial_segments,
    const int stride,
    FastDivmod stride_warped_fastdv,
    const int padding_idx) {
  using accscalar_t = acc_type<scalar_t, true>;
  using vec_dtype = VecType<scalar_t, vlen * sizeof(scalar_t) * 8>;
  using vec_acc_dtype = VecType<accscalar_t, vlen * sizeof(accscalar_t) * 8>;

  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t id, feature_offset; // quotient, remainder
  stride_warped_fastdv(id, feature_offset, gid);
  uint32_t feature_offset_vlen = feature_offset * vlen;
  if (feature_offset_vlen >= stride) {
    return;
  }
  if (id >= num_of_segments) {
    return;
  }
  const int idx_start = partials_per_segment_offset[id];
  const int idx_end = (id == num_of_segments - 1)
      ? num_of_partial_segments
      : partials_per_segment_offset[id + 1];
  index_t target_row = idx[segment_offsets[id]];
  if (feature_offset_vlen + vlen <= stride) {
    vec_acc_dtype weight_acc;
    vec_dtype weight;
    for (int idx = idx_start; idx < idx_end; idx++) {
      vec_acc_dtype reg_dw =
          vec_acc_dtype::load(dw_segments, idx * stride + feature_offset_vlen);
#pragma unroll
      for (int k = 0; k < vlen; k++) {
        weight_acc.val_.elem[k] += (accscalar_t)reg_dw.val_.elem[k];
      }
    }
#pragma unroll
    for (int k = 0; k < vlen; k++) {
      weight.val_.elem[k] += (scalar_t)weight_acc.val_.elem[k];
    }
    if (target_row != padding_idx) {
      // #pragma unroll
      // for (int k = 0; k < vlen; k++) {
      //   dw[target_row * stride + feature_offset_vlen + k] =
      //   weight.val_.elem[k];
      // }
      vec_dtype::store(dw, target_row * stride + feature_offset_vlen, weight);
    }
  } else {
    while (feature_offset_vlen < stride) {
      acc_type<scalar_t, true> weight = 0;
      for (int idx = idx_start; idx < idx_end; idx++) {
        weight += dw_segments[idx * stride + feature_offset_vlen];
      }
      if (target_row != padding_idx) {
        dw[target_row * stride + feature_offset_vlen] = weight;
      }
      feature_offset_vlen++;
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void SumAndScatter(
    scalar_t* dw,
    const acc_type<scalar_t, true>* dw_segments,
    const index_t* idx,
    const index_t* segment_offsets,
    const index_t* partials_per_segment_offset,
    const int num_of_segments,
    const int num_of_partial_segments,
    const int stride,
    FastDivmod stride_warped_fastdv,
    const int padding_idx) {
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t id, feature_offset; // quotient, remainder
  stride_warped_fastdv(id, feature_offset, gid);
  if (feature_offset >= stride) {
    return;
  }
  if (id >= num_of_segments) {
    return;
  }
  const int idx_start = partials_per_segment_offset[id];
  const int idx_end = (id == num_of_segments - 1)
      ? num_of_partial_segments
      : partials_per_segment_offset[id + 1];
  acc_type<scalar_t, true> weight = 0;
  for (int idx = idx_start; idx < idx_end; idx++) {
    weight += dw_segments[idx * stride + feature_offset];
  }
  index_t target_row = idx[segment_offsets[id]];
  if (target_row != padding_idx) {
    dw[target_row * stride + feature_offset] = weight;
  }
}
} // namespace

Tensor EmbeddingBackwardMUSAKernel(
    const Tensor& grad,
    const Tensor& orig_indices,
    const Tensor& sorted_indices,
    int64_t num_weights,
    int padding_idx) {
  auto stream = at::musa::getCurrentMUSAStream();
  const ptrdiff_t numel = sorted_indices.numel();
  Tensor grad_weight = at::zeros({num_weights, grad.size(-1)}, grad.options());

  int tbl_h = grad_weight.size(0);
  int tbl_w = grad_weight.size(1);

  // Compute the number of segments and segment offsets
  auto segment_offsets = at::empty({numel}, orig_indices.options());
  int32_t num_of_segments = 0;

  AT_DISPATCH_INDEX_TYPES(
      orig_indices.scalar_type(), "EmbeddingBackwardMUSAKernel", [&]() {
        num_of_segments = EmbeddingBackwardMUSAKernelUniqueByKey<index_t>(
            sorted_indices, segment_offsets);
      });

  // Split the segments up into size of `NROWS_PER_THREAD`
  // Compute the number partial-segments per segment
  int64_t max_segment = std::min<int64_t>(numel, num_weights);
  auto partials_per_segment = at::empty({max_segment}, orig_indices.options());
  int max_blocks = 64;
  if (at::musa::getMUSAArch() == 110) {
    max_blocks = 16;
  } else if (at::musa::getMUSAArch() == 210) {
    max_blocks = 32;
  }
  uint32_t threads = 1024;
  uint32_t blocks =
      std::min(ceil_div(numel, threads), static_cast<int64_t>(max_blocks));

  AT_DISPATCH_INDEX_TYPES(
      orig_indices.scalar_type(), "EmbeddingBackwardMUSAKernel", [&]() {
        {
          PartialsPerSegment<<<blocks, threads, 0, stream>>>(
              partials_per_segment.data_ptr<index_t>(),
              segment_offsets.data_ptr<index_t>(),
              num_of_segments,
              numel);
          C10_MUSA_KERNEL_LAUNCH_CHECK();
        }

        auto partials_per_segment_offset =
            at::empty({max_segment}, orig_indices.options());

        at::musa::cub::exclusive_sum(
            partials_per_segment.data_ptr<index_t>(),
            partials_per_segment_offset.data_ptr<index_t>(),
            max_segment);

        // The total number of partial-segments is the sum of
        // `partials_per_segment_offset`
        auto num_of_partial_segments_tensor =
            at::empty({}, grad.options().dtype(kLong));
        int64_t num_of_partial_segments = 0;

        auto max_partial_segment = numel / NROWS_PER_THREAD + max_segment;

        auto partial_segment_offset =
            at::empty({max_partial_segment}, orig_indices.options());
        int64_t* num_of_partial_segments_ptr =
            num_of_partial_segments_tensor.data_ptr<int64_t>();

        {
          PartialsSegmentOffset<<<blocks, threads, 0, stream>>>(
              num_of_partial_segments_ptr,
              partial_segment_offset.data_ptr<index_t>(),
              partials_per_segment.data_ptr<index_t>(),
              partials_per_segment_offset.data_ptr<index_t>(),
              segment_offsets.data_ptr<index_t>(),
              num_of_segments);
          C10_MUSA_KERNEL_LAUNCH_CHECK();
        }

        musaMemcpyAsync(
            &num_of_partial_segments,
            static_cast<void*>(num_of_partial_segments_ptr),
            sizeof(int64_t),
            static_cast<musaMemcpyKind>(musaMemcpyDeviceToHost),
            stream);
        musaStreamSynchronize(stream);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            grad.scalar_type(),
            "embedding_backward_musa",
            [&] {
              using partial_weight_t = acc_type<scalar_t, true>;
              TensorOptions op;
              if (grad.dtype() == at::kHalf || grad.dtype() == at::kBFloat16) {
                op = grad.options().dtype(at::kFloat);
              } else {
                op = grad.options();
              }
              auto grad_weight_per_segment =
                  at::empty({num_of_partial_segments, tbl_w}, op);

              const int warp_size = at::musa::warp_size();
              int64_t vlen = std::min(
                  at::musa::can_vectorize_up_to<scalar_t>(
                      (char*)grad.data_ptr()),
                  at::musa::can_vectorize_up_to<partial_weight_t>(
                      (char*)grad_weight_per_segment.data_ptr()));
              // bool can_vectorize = vlen > 1 && (tbl_w % vlen == 0);
              bool can_vectorize = vlen > 1;
              vlen = can_vectorize ? vlen : 1;
              const int stride_warped =
                  ceil_div(tbl_w, warp_size * vlen) * warp_size;
              const int block = std::min(stride_warped, 512);
              const int grid =
                  ceil_div(max_partial_segment * stride_warped, block);
              const int grid2 = ceil_div(max_segment * stride_warped, block);

              // 1. Compute the grad of each partial-segment
              // 2. Sum all the partial-sums and scatter them into `grad_weight`
              auto stride_warped_fastdv = FastDivmod((uint32_t)stride_warped);
#define VEC_CASE(_VLEN)                                      \
  case (_VLEN):                                              \
    ComputeDwSegmentVector<scalar_t, index_t, _VLEN>         \
        <<<grid, block, 0, stream>>>(                        \
            static_cast<partial_weight_t*>(                  \
                grad_weight_per_segment.data_ptr()),         \
            static_cast<scalar_t*>(grad.data_ptr()),         \
            orig_indices.data_ptr<index_t>(),                \
            partial_segment_offset.data_ptr<index_t>(),      \
            num_of_partial_segments,                         \
            numel,                                           \
            tbl_w,                                           \
            stride_warped_fastdv);                           \
    C10_MUSA_KERNEL_LAUNCH_CHECK();                          \
                                                             \
    SumAndScatterVector<scalar_t, index_t, _VLEN>            \
        <<<grid2, block, 0, stream>>>(                       \
            static_cast<scalar_t*>(grad_weight.data_ptr()),  \
            static_cast<partial_weight_t*>(                  \
                grad_weight_per_segment.data_ptr()),         \
            sorted_indices.data_ptr<index_t>(),              \
            segment_offsets.data_ptr<index_t>(),             \
            partials_per_segment_offset.data_ptr<index_t>(), \
            num_of_segments,                                 \
            num_of_partial_segments,                         \
            tbl_w,                                           \
            stride_warped_fastdv,                            \
            padding_idx);                                    \
    C10_MUSA_KERNEL_LAUNCH_CHECK();                          \
    break
              if (can_vectorize) {
                switch (vlen) {
                  VEC_CASE(2);
                  VEC_CASE(4);
                  default:
                    TORCH_CHECK(
                        false,
                        "Not supported vectorized length ",
                        vlen,
                        "in EmbeddingDenseBwdMUSA Kernel");
                }
              } else {
                ComputeDwSegment<<<grid, block, 0, stream>>>(
                    static_cast<partial_weight_t*>(
                        grad_weight_per_segment.data_ptr()),
                    static_cast<scalar_t*>(grad.data_ptr()),
                    orig_indices.data_ptr<index_t>(),
                    partial_segment_offset.data_ptr<index_t>(),
                    num_of_partial_segments,
                    numel,
                    tbl_w,
                    stride_warped_fastdv);
                C10_MUSA_KERNEL_LAUNCH_CHECK();

                SumAndScatter<<<grid2, block, 0, stream>>>(
                    static_cast<scalar_t*>(grad_weight.data_ptr()),
                    static_cast<partial_weight_t*>(
                        grad_weight_per_segment.data_ptr()),
                    sorted_indices.data_ptr<index_t>(),
                    segment_offsets.data_ptr<index_t>(),
                    partials_per_segment_offset.data_ptr<index_t>(),
                    num_of_segments,
                    num_of_partial_segments,
                    tbl_w,
                    stride_warped_fastdv,
                    padding_idx);
                C10_MUSA_KERNEL_LAUNCH_CHECK();
              }
            });
      });
#undef VEC_CASE
  return grad_weight;
}

} // namespace native
} // namespace at
