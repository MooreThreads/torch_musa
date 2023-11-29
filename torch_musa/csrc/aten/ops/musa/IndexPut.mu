#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Array.h>
#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>

#include <musa_bf16.h>
#include <musa_fp16.h>
#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/musa/MUSAAtomic.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/ops/TensorShape.h"
#include "torch_musa/csrc/aten/ops/musa/IndexUtils.muh"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace native {

typedef __half float16_t;
typedef __mt_bfloat16 bfloat16_t;

namespace {
constexpr int MAX_DIM = kMaxDim;

template <
    typename DstDtype,
    typename IndexDtype,
    typename SrcDtype = DstDtype,
    typename OffsetDType,
    int OutNdim,
    int MaxDim = MAX_DIM>
__global__ void index_put_kernel(
    DstDtype* out_ptr,
    SrcDtype* value_ptr,
    at::detail::Array<void*, MaxDim> inds,
    const int nthreads,
    const int out_ndim,
    const int task_ndim,
    const int start_of_remain_idx,
    const int num_of_null_in_front,
    const int not_null_idx_cnt,
    const unsigned char accumulate,
    at::detail::Array<int, MaxDim> task_shapes,
    at::detail::Array<int, MaxDim> out_shapes,
    at::detail::Array<int, MaxDim> out_strides,
    at::detail::Array<int, MaxDim> value_strides,
    at::detail::Array<int, MaxDim * MaxDim> ind_strides) {
  for (OffsetDType idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nthreads;
       idx += blockDim.x * gridDim.x) {
    OffsetDType reminder = idx;
    OffsetDType index[MaxDim]; // index in task tensor
    OffsetDType in_index[MaxDim] = {0}; // index in out tensor

#pragma unroll
    for (int i = task_ndim - 1; i >= 0; i--) {
      index[i] = reminder % task_shapes[i];
      reminder /= task_shapes[i];
    }
#pragma unroll
    for (int i = 0; i < OutNdim; ++i) {
      in_index[i] = index[i];
    }

    int in_p = num_of_null_in_front;
    OffsetDType indices_offset;
    for (int i = 0; i < not_null_idx_cnt; ++i) {
      indices_offset = 0;
      int64_t* indices_ptr = static_cast<int64_t*>(inds[i]);
#pragma unroll
      for (int j = OutNdim - 1; j >= 0; j--) {
        indices_offset += index[j] * ind_strides[i * MaxDim + j];
      }
      in_index[in_p++] = static_cast<OffsetDType>(indices_ptr[indices_offset]);
    }

    for (int i = in_p, j = start_of_remain_idx; in_p < out_ndim && j < OutNdim;
         ++i, ++j) {
      in_index[i] = index[j];
    }

    OffsetDType in_offset = 0;
#pragma unroll
    for (int i = 0; i < MaxDim; ++i) {
      in_offset += in_index[i] * out_strides[i];
    }

    OffsetDType value_offset = 0;
#pragma unroll
    for (int i = out_ndim - 1; i >= 0; --i) {
      value_offset += (index[i] * value_strides[i]);
    }

    if (accumulate == 1) {
      at::musa::gpuAtomicAdd(&out_ptr[in_offset], value_ptr[value_offset]);
    } else {
      out_ptr[in_offset] = value_ptr[value_offset];
    }
  }
}

#define GEN_FUNCTION(_SRC, _IDX, _OFFSET_DTYPE, _OUT_NDIM)       \
  [](const Tensor& out,                                          \
     const Tensor& value,                                        \
     at::detail::Array<void*, MAX_DIM>& inds,                    \
     const int nthreads,                                         \
     const int out_ndim,                                         \
     const int task_ndim,                                        \
     const int start_of_remain_idx,                              \
     const int num_of_null_in_front,                             \
     const int not_null_idx_cnt,                                 \
     const unsigned char accumulate,                             \
     at::detail::Array<int, MAX_DIM>& task_shapes,               \
     at::detail::Array<int, MAX_DIM>& out_shapes,                \
     at::detail::Array<int, MAX_DIM>& out_strides,               \
     at::detail::Array<int, MAX_DIM>& value_strides,             \
     at::detail::Array<int, MAX_DIM * MAX_DIM>& ind_strides,     \
     const int nr_block,                                         \
     const int thread_per_block) {                               \
    auto stream = c10::musa::getCurrentMUSAStream();             \
    index_put_kernel<_SRC, _IDX, _SRC, _OFFSET_DTYPE, _OUT_NDIM> \
        <<<nr_block, thread_per_block, 0, stream>>>(             \
            static_cast<_SRC*>(out.data_ptr()),                  \
            static_cast<_SRC*>(value.data_ptr()),                \
            inds,                                                \
            nthreads,                                            \
            out_ndim,                                            \
            task_ndim,                                           \
            start_of_remain_idx,                                 \
            num_of_null_in_front,                                \
            not_null_idx_cnt,                                    \
            accumulate,                                          \
            task_shapes,                                         \
            out_shapes,                                          \
            out_strides,                                         \
            value_strides,                                       \
            ind_strides);                                        \
  }

#define REGISTER_KERNEL(_DTYPE_ENUM, _OUT_NDIM, _CTYPE)      \
  index_put_kernels[0][(int)_DTYPE_ENUM][0][_OUT_NDIM - 1] = \
      GEN_FUNCTION(_CTYPE, int32_t, int32_t, _OUT_NDIM);     \
  index_put_kernels[1][(int)_DTYPE_ENUM][0][_OUT_NDIM - 1] = \
      GEN_FUNCTION(_CTYPE, int64_t, int32_t, _OUT_NDIM);     \
  index_put_kernels[2][(int)_DTYPE_ENUM][0][_OUT_NDIM - 1] = \
      GEN_FUNCTION(_CTYPE, bool, int32_t, _OUT_NDIM);        \
  index_put_kernels[0][(int)_DTYPE_ENUM][1][_OUT_NDIM - 1] = \
      GEN_FUNCTION(_CTYPE, int32_t, int64_t, _OUT_NDIM);     \
  index_put_kernels[1][(int)_DTYPE_ENUM][1][_OUT_NDIM - 1] = \
      GEN_FUNCTION(_CTYPE, int64_t, int64_t, _OUT_NDIM);     \
  index_put_kernels[2][(int)_DTYPE_ENUM][1][_OUT_NDIM - 1] = \
      GEN_FUNCTION(_CTYPE, bool, int64_t, _OUT_NDIM);

struct KernelTable {
  using KernelFunc = std::function<void(
      const Tensor&,
      const Tensor&,
      at::detail::Array<void*, MAX_DIM>&,
      const int,
      const int,
      const int,
      const int,
      const int,
      const int,
      const unsigned char,
      at::detail::Array<int, MAX_DIM>&,
      at::detail::Array<int, MAX_DIM>&,
      at::detail::Array<int, MAX_DIM>&,
      at::detail::Array<int, MAX_DIM>&,
      at::detail::Array<int, MAX_DIM * MAX_DIM>&,
      const int,
      const int)>;

#define REGISTER_KERNEL_DTYPE(_OUT_NDIM)                           \
  REGISTER_KERNEL(at::ScalarType::BFloat16, _OUT_NDIM, bfloat16_t) \
  REGISTER_KERNEL(at::ScalarType::Half, _OUT_NDIM, float16_t)      \
  REGISTER_KERNEL(at::ScalarType::Float, _OUT_NDIM, float)         \
  REGISTER_KERNEL(at::ScalarType::Double, _OUT_NDIM, double)       \
  REGISTER_KERNEL(at::ScalarType::Int, _OUT_NDIM, int32_t)         \
  REGISTER_KERNEL(at::ScalarType::Long, _OUT_NDIM, int64_t)

  KernelTable() {
    REGISTER_KERNEL_DTYPE(1)
    REGISTER_KERNEL_DTYPE(2)
    REGISTER_KERNEL_DTYPE(3)
    REGISTER_KERNEL_DTYPE(4)
    REGISTER_KERNEL_DTYPE(5)
    REGISTER_KERNEL_DTYPE(6)
    REGISTER_KERNEL_DTYPE(7)
    REGISTER_KERNEL_DTYPE(8)
  }
#undef REGISTER_KERNEL_DTYPE

  void launch(
      const Tensor& out,
      const Tensor& value,
      at::detail::Array<void*, MAX_DIM>& inds,
      const int nthreads,
      const int out_ndim,
      const int task_ndim,
      const int start_of_remain_idx,
      const int num_of_null_in_front,
      const int not_null_idx_cnt,
      const unsigned char accumulate,
      at::detail::Array<int, MAX_DIM>& task_shapes,
      at::detail::Array<int, MAX_DIM>& out_shapes,
      at::detail::Array<int, MAX_DIM>& out_strides,
      at::detail::Array<int, MAX_DIM>& value_strides,
      at::detail::Array<int, MAX_DIM * MAX_DIM>& ind_strides,
      at::ScalarType out_dtype,
      at::ScalarType index_dtype,
      const int nr_block,
      const int thread_per_block) {
    int offset_types_tag =
        (out.numel() < std::numeric_limits<std::int32_t>::max()) &&
            (value.numel() < std::numeric_limits<std::int32_t>::max())
        ? 0
        : 1;
    int indices_tag = (index_dtype == at::ScalarType::Int)
        ? 0
        : (index_dtype == at::ScalarType::Long ? 1 : 2);
    auto& func = index_put_kernels[indices_tag][int(out_dtype)]
                                  [offset_types_tag][task_ndim - 1];

    if (func) {
      func(
          out,
          value,
          inds,
          nthreads,
          out_ndim,
          task_ndim,
          start_of_remain_idx,
          num_of_null_in_front,
          not_null_idx_cnt,
          accumulate,
          task_shapes,
          out_shapes,
          out_strides,
          value_strides,
          ind_strides,
          nr_block,
          thread_per_block);
    } else {
      TORCH_CHECK(false, "IndexPut nsupported!");
    }
  }

  static constexpr int indices_types = 3; // int32 & int64 & bool
  static constexpr int nr_dtype = (int)at::ScalarType::NumOptions;
  static constexpr int out_ndim_num = 8;
  static constexpr int offset_types = 2; // int32 & int64
  KernelFunc index_put_kernels[indices_types][nr_dtype][offset_types]
                              [out_ndim_num];
};

} // namespace

void IndexPutRun(
    Tensor& out,
    const std::vector<Tensor>& indices,
    const Tensor& value,
    const bool accumulate) {
  TORCH_CHECK(
      (out.scalar_type() == at::ScalarType::Float) ||
          (out.scalar_type() == at::ScalarType::Long) ||
          (out.scalar_type() == at::ScalarType::Half) ||
          (out.scalar_type() == at::ScalarType::Double) ||
          (out.scalar_type() == at::ScalarType::Int) ||
          (out.scalar_type() == at::ScalarType::BFloat16),
      "IndexPut only support input dtype bf16, fp16/32/64, int32/64, got ",
      out.scalar_type());

  at::musa::muHandle& h = at::GetMudnnHandle();
  int indices_num = indices.size();
  const int out_ndim = out.dim();

  at::detail::Array<int, MAX_DIM> bcast_indice_shape;
  int bcast_indice_ndim = 0;
  IndicesBroadCast(indices_num, indices, bcast_indice_shape, bcast_indice_ndim);

  int not_null_idx_cnt = 0;
  at::detail::Array<void*, MAX_DIM> inds_addr;
  for (int i = 0, j = 0; i < indices_num; ++i) {
    if (indices[i].numel() != 0) {
      inds_addr[j++] = indices[i].data_ptr();
      not_null_idx_cnt++;
    }
  }

  if (not_null_idx_cnt == 0) {
    if (value.dim() == out.dim()) {
      out.copy_(value);
      return;
    } else {
      out = at::empty_like(out, out.options().device(kMUSA)).fill_(value);
      return;
    }
  }

  const int task_ndim = bcast_indice_ndim + out_ndim - not_null_idx_cnt;

  std::vector<Tensor> not_null_indices(not_null_idx_cnt);
  for (int i = 0, j = 0; i < indices_num; ++i) {
    if (indices[i].numel() != 0) {
      not_null_indices[j++] = indices[i];
    }
  }

  // calculate task shape of related indices
  std::vector<int> task_shape_vec;
  bool has_defined = false;
  for (int i = 0; i < indices_num; ++i) {
    if (indices[i].numel() > 0) {
      if (!has_defined) {
        auto indice_size = indices[i].sizes().vec();
        task_shape_vec.insert(
            task_shape_vec.end(), indice_size.begin(), indice_size.end());
        has_defined = true;
      }
    } else {
      task_shape_vec.emplace_back(out.size(i));
    }
  }

  bool has_contiguous_subspace = HasContiguousSubspace(indices_num, indices);
  // [:, :, a, b, :, c], num_of_null_in_front = 2
  int num_of_null_in_front = 0;
  if (has_contiguous_subspace) {
    for (int i = 0; i < indices_num; ++i) {
      if (indices[i].numel() == 0) {
        num_of_null_in_front++;
      } else {
        break;
      }
    }
  }

  // calculate strides of indice/value/out
  at::detail::Array<int, MAX_DIM> output_shape;
  at::detail::Array<int, MAX_DIM> task_shape;
  std::copy(
      out.sizes().data(),
      out.sizes().data() + out_ndim,
      std::begin(output_shape.data));
  std::copy(
      task_shape_vec.begin(),
      task_shape_vec.end(),
      std::begin(task_shape.data));
  ValueBroadcastableCheck(value, task_shape, task_ndim);
  at::detail::Array<int, MAX_DIM * MAX_DIM> strides_ind;
  at::detail::Array<int, MAX_DIM * MAX_DIM> shapes_ind;
  at::detail::Array<int, MAX_DIM> strides_value;
  at::detail::Array<int, MAX_DIM> shapes_value;
  at::detail::Array<int, MAX_DIM> strides_out;
  IndicesStrides(
      strides_ind,
      shapes_ind,
      bcast_indice_shape,
      task_shape,
      not_null_idx_cnt,
      not_null_indices,
      bcast_indice_ndim,
      task_ndim,
      num_of_null_in_front);
  ValueStrides(strides_value, shapes_value, task_shape, value, task_ndim);
  InputStrides(strides_out, out, indices_num, indices, has_contiguous_subspace);

  int nthreads = 1;
  for (int i = 0; i < task_ndim; ++i) {
    nthreads *= task_shape[i];
  }

  // device info
  musaDeviceProp device_prop;
  int device_id = h.GetDeviceId();
  TORCH_CHECK(
      musaSuccess == musaGetDeviceProperties(&device_prop, device_id),
      "musaGetDeviceProperties error");
  const int mp_num = device_prop.multiProcessorCount;
  const int block_size = 1024;
  const int block_num =
      std::min(at::musa::ceil_div(nthreads, block_size), mp_num * 4);

  static KernelTable kernel_pack;

  kernel_pack.launch(
      out,
      value,
      inds_addr,
      nthreads,
      out_ndim,
      task_ndim,
      bcast_indice_ndim,
      num_of_null_in_front,
      not_null_idx_cnt,
      accumulate,
      task_shape,
      output_shape,
      strides_out,
      strides_value,
      strides_ind,
      out.scalar_type(),
      not_null_indices[0].scalar_type(),
      block_num,
      block_size);
}

REGISTER_MUSA_DISPATCH(indexput_stub, &IndexPutRun);

} // namespace native
} // namespace at
