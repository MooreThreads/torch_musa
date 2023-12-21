#include <ATen/ATen.h>
#include <ATen/core/Array.h>
#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/ops/TensorShape.h"
#include "torch_musa/csrc/aten/ops/musa/IndexUtils.muh"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <algorithm>

namespace at {
namespace native {

namespace {
constexpr int MAX_DIM = kMaxDim;

template <
    typename DstDtype,
    typename IndicesDType,
    typename SrcDtype = DstDtype,
    typename OffsetDType,
    int OutNdim,
    int MaxDim = MAX_DIM>
__global__ void IndexKernel(
    DstDtype* out_ptr,
    SrcDtype* in_ptr,
    at::detail::Array<void*, MaxDim> indices,
    const int elements,
    const int out_ndim,
    const int in_ndim,
    const int start_of_remain_idx,
    const int not_null_idx_cnt,
    const int num_of_null_in_front,
    at::detail::Array<int, MaxDim> shapes_out,
    at::detail::Array<int, MaxDim> strides_in,
    at::detail::Array<int, MaxDim * MaxDim> strides_indices) {
  for (OffsetDType idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements;
       idx += blockDim.x * gridDim.x) {
    OffsetDType reminder = idx;
    OffsetDType index[MaxDim];
    OffsetDType in_index[MaxDim] = {0};

#pragma unroll
    for (int i = out_ndim - 1; i >= 0; i--) {
      index[i] = reminder % shapes_out[i];
      reminder /= shapes_out[i];
    }
#pragma unroll
    for (int i = 0; i < OutNdim; ++i) { // i < num_of_null_in_front
      in_index[i] = index[i];
    }

    int in_p = num_of_null_in_front;
    OffsetDType indices_offset;
    for (int j = 0; j < not_null_idx_cnt; ++j) {
      indices_offset = 0;
      int64_t* indices_ptr = static_cast<int64_t*>(indices[j]);
#pragma unroll
      for (int k = OutNdim - 1; k >= 0; --k) {
        indices_offset += index[k] * strides_indices[j * MaxDim + k];
      }
      in_index[in_p++] = static_cast<OffsetDType>(indices_ptr[indices_offset]);
    }

    for (int i = in_p, j = start_of_remain_idx; in_p < in_ndim && j < OutNdim;
         ++i, ++j) {
      in_index[i] = index[j];
    }

    OffsetDType in_offset = 0;
#pragma unroll
    for (int i = 0; i < MaxDim; ++i) {
      in_offset += in_index[i] * strides_in[i];
    }

    out_ptr[idx] = in_ptr[in_offset];
  }
}

#define GEN_FUNCTION(_SRC, _IDX, _OFFSET_DTYPE, _OUT_NDIM)       \
  [](const Tensor& out,                                          \
     const Tensor& in,                                           \
     at::detail::Array<void*, MAX_DIM>& indices,                 \
     const int elements,                                         \
     const int out_ndim,                                         \
     const int in_ndim,                                          \
     const int start_of_remain_idx,                              \
     const int not_null_idx_cnt,                                 \
     const int num_of_null_in_front,                             \
     at::detail::Array<int, MAX_DIM>& shapes_out,                \
     at::detail::Array<int, MAX_DIM>& strides_in,                \
     at::detail::Array<int, MAX_DIM * MAX_DIM>& strides_indices, \
     const int nr_block,                                         \
     const int thread_per_block) {                               \
    auto stream = c10::musa::getCurrentMUSAStream();             \
    IndexKernel<_SRC, _IDX, _SRC, _OFFSET_DTYPE, _OUT_NDIM>      \
        <<<nr_block, thread_per_block, 0, stream>>>(             \
            static_cast<_SRC*>(out.data_ptr()),                  \
            static_cast<_SRC*>(in.data_ptr()),                   \
            indices,                                             \
            elements,                                            \
            out_ndim,                                            \
            in_ndim,                                             \
            start_of_remain_idx,                                 \
            not_null_idx_cnt,                                    \
            num_of_null_in_front,                                \
            shapes_out,                                          \
            strides_in,                                          \
            strides_indices);                                    \
  }

#define REGISTER_KERNEL(_DTYPE_ENUM, _OUT_NDIM, _CTYPE)  \
  index_kernels[0][(int)_DTYPE_ENUM][0][_OUT_NDIM - 1] = \
      GEN_FUNCTION(_CTYPE, int32_t, int32_t, _OUT_NDIM); \
  index_kernels[1][(int)_DTYPE_ENUM][0][_OUT_NDIM - 1] = \
      GEN_FUNCTION(_CTYPE, int64_t, int32_t, _OUT_NDIM); \
  index_kernels[2][(int)_DTYPE_ENUM][0][_OUT_NDIM - 1] = \
      GEN_FUNCTION(_CTYPE, bool, int32_t, _OUT_NDIM);    \
  index_kernels[0][(int)_DTYPE_ENUM][1][_OUT_NDIM - 1] = \
      GEN_FUNCTION(_CTYPE, int32_t, int64_t, _OUT_NDIM); \
  index_kernels[1][(int)_DTYPE_ENUM][1][_OUT_NDIM - 1] = \
      GEN_FUNCTION(_CTYPE, int64_t, int64_t, _OUT_NDIM); \
  index_kernels[2][(int)_DTYPE_ENUM][1][_OUT_NDIM - 1] = \
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
      at::detail::Array<int, MAX_DIM>&,
      at::detail::Array<int, MAX_DIM>&,
      at::detail::Array<int, MAX_DIM * MAX_DIM>&,
      const int,
      const int)>;

#define REGISTER_KERNEL_DTYPE(_OUT_NDIM)                           \
  REGISTER_KERNEL(at::ScalarType::BFloat16, _OUT_NDIM, bfloat16_t) \
  REGISTER_KERNEL(at::ScalarType::Char, _OUT_NDIM, int8_t)         \
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
      const Tensor& in,
      at::detail::Array<void*, MAX_DIM>& indices,
      const int elements,
      const int out_ndim,
      const int in_ndim,
      const int indices_broadcast_ndim,
      const int not_null_idx_cnt,
      const int num_of_null_in_front,
      at::detail::Array<int, MAX_DIM>& shapes_out,
      at::detail::Array<int, MAX_DIM>& strides_in,
      at::detail::Array<int, MAX_DIM * MAX_DIM>& strides_indices,
      at::ScalarType in_dtype,
      at::ScalarType indices_dtype,
      const int nr_block,
      const int thread_per_block) {
    int offset_types_tag =
        (out.numel() < std::numeric_limits<std::int32_t>::max()) &&
            (in.numel() < std::numeric_limits<std::int32_t>::max())
        ? 0
        : 1;
    int indices_tag = (indices_dtype == at::ScalarType::Int)
        ? 0
        : (indices_dtype == at::ScalarType::Long ? 1 : 2);
    auto& func = index_kernels[indices_tag][int(in_dtype)][offset_types_tag]
                              [out_ndim - 1];

    if (func) {
      func(
          out,
          in,
          indices,
          elements,
          out_ndim,
          in_ndim,
          indices_broadcast_ndim + num_of_null_in_front,
          not_null_idx_cnt,
          num_of_null_in_front,
          shapes_out,
          strides_in,
          strides_indices,
          nr_block,
          thread_per_block);
    } else {
      TORCH_CHECK(false, "Index unsupported!");
    }
  }

  static constexpr int indices_types = 3; // int32 & int64 & bool
  static constexpr int nr_dtype = (int)at::ScalarType::NumOptions;
  static constexpr int out_ndim_num = 8;
  static constexpr int offset_types = 2; // int32 & int64
  KernelFunc index_kernels[indices_types][nr_dtype][offset_types][out_ndim_num];
};
} // namespace

void IndexRun(
    Tensor& out,
    int indices_num,
    const std::vector<Tensor>& indices,
    const Tensor& in) {
  TORCH_CHECK(
      indices_num <= in.dim(),
      "Too many indices for tensor of dimension ",
      in.dim(),
      ", got ",
      indices_num);
  TORCH_CHECK(
      (in.scalar_type() == at::ScalarType::Float) ||
          (in.scalar_type() == at::ScalarType::Long) ||
          (in.scalar_type() == at::ScalarType::Half) ||
          (in.scalar_type() == at::ScalarType::Double) ||
          (in.scalar_type() == at::ScalarType::Int) ||
          (in.scalar_type() == at::ScalarType::Char) ||
          (in.scalar_type() == at::ScalarType::BFloat16),
      "Index only support input dtype bf16, fp16/32/64, int32/64, got ",
      in.scalar_type());

  at::musa::muHandle& h = GetMudnnHandle();
  at::detail::Array<int, MAX_DIM> indices_broadcast_shape;
  int indices_broadcast_ndim = 0;
  IndicesBroadCast(
      indices_num, indices, indices_broadcast_shape, indices_broadcast_ndim);

  const int out_ndim = out.dim();
  int not_null_idx_cnt = 0;
  at::detail::Array<void*, MAX_DIM> indices_addr;
  for (int i = 0, j = 0; i < indices_num; ++i) {
    if (indices[i].numel() > 0) {
      indices_addr[j++] = indices[i].data_ptr();
      not_null_idx_cnt++;
    }
  }

  std::vector<Tensor> not_null_indices(not_null_idx_cnt);
  for (int i = 0, j = 0; i < indices_num; ++i) {
    if (indices[i].numel() > 0) {
      not_null_indices[j++] = indices[i];
    }
  }

  if (not_null_idx_cnt == 0) {
    out.copy_(in);
    return;
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

  // calculate strides of out/indices/in
  at::detail::Array<int, MAX_DIM> output_shape;
  std::copy(
      out.sizes().data(),
      out.sizes().data() + out_ndim,
      std::begin(output_shape.data));
  at::detail::Array<int, MAX_DIM * MAX_DIM> strides_indices;
  at::detail::Array<int, MAX_DIM * MAX_DIM> shapes_indices;
  at::detail::Array<int, MAX_DIM> strides_in;
  IndicesStrides(
      strides_indices,
      shapes_indices,
      indices_broadcast_shape,
      output_shape,
      not_null_idx_cnt,
      not_null_indices,
      indices_broadcast_ndim,
      out_ndim,
      num_of_null_in_front);
  InputStrides(strides_in, in, indices_num, indices, has_contiguous_subspace);

  // device info
  musaDeviceProp device_prop;
  int device_id = h.GetDeviceId();
  TORCH_CHECK(
      musaSuccess == musaGetDeviceProperties(&device_prop, device_id),
      "musaGetDeviceProperties error");
  const int mp_num = device_prop.multiProcessorCount;
  const int elements = out.numel();
  const int nr_threads = 1024;
  const int nr_blocks =
      std::min(at::musa::ceil_div(elements, nr_threads), mp_num * 4);

  KernelTable kernel_index;
  kernel_index.launch(
      out,
      in,
      indices_addr,
      elements,
      out_ndim,
      in.dim(),
      indices_broadcast_ndim,
      not_null_idx_cnt,
      num_of_null_in_front,
      output_shape,
      strides_in,
      strides_indices,
      in.scalar_type(),
      indices[0].scalar_type(),
      nr_blocks,
      nr_threads);
}

REGISTER_MUSA_DISPATCH(indexes_stub, &IndexRun);

} // namespace native
} // namespace at
