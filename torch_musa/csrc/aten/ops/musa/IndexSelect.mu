#include <ATen/ATen.h>
#include <ATen/core/Array.h>
#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/ops/TensorShape.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <algorithm>

namespace at {
namespace native {

namespace {

template <
    typename SrcDtype,
    typename IndexDtype,
    bool Aligned = true,
    int iobit = 128>
__global__ void IndexSelectVectorKernel(
    SrcDtype* out_ptr,
    IndexDtype* index_ptr,
    SrcDtype* in_ptr,
    const int64_t r0,
    const int64_t s0,
    int64_t num_indices,
    const int64_t elements,
    const uint32_t tail) {
  // iobit is used to choose length of vector load, which is following muDNN
  constexpr int bits_of_byte = 8;
  constexpr int vlen_min = iobit / sizeof(SrcDtype) / bits_of_byte;
  constexpr int max_load_vlen =
      sizeof(SrcDtype) <= 4 ? (sizeof(SrcDtype) <= 2 ? 8 : 4) : 2;
  constexpr int vlen = vlen_min > max_load_vlen ? max_load_vlen : vlen_min;
  using SrcVec =
      at::musa::VecType<SrcDtype, vlen * sizeof(SrcDtype) * bits_of_byte>;

  int64_t index_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t index_selected = index_ptr[index_idx];
  int64_t idx_x = (blockIdx.x * blockDim.x + threadIdx.x);

  if constexpr (Aligned) {
    if (index_idx < num_indices && idx_x < (s0 / vlen)) {
      idx_x *= vlen;
      int64_t src_offset = (blockIdx.z * r0 + index_selected) * s0 + idx_x;
      int64_t dst_offset = (blockIdx.z * num_indices + index_idx) * s0 + idx_x;
      SrcVec vec_in = SrcVec::load(in_ptr, src_offset);
      SrcVec::store(out_ptr, dst_offset, vec_in);
    }
  } else {
    if (index_idx < num_indices) {
      int sv = s0 / vlen;
      if (idx_x < sv) {
        idx_x *= vlen;
        int64_t src_offset = (blockIdx.z * r0 + index_selected) * s0 + idx_x;
        int64_t dst_offset =
            (blockIdx.z * num_indices + index_idx) * s0 + idx_x;
        SrcVec vec_in = SrcVec::load(in_ptr, src_offset);
        SrcVec::store(out_ptr, dst_offset, vec_in);
      } else if (idx_x == sv && tail > 0) {
        idx_x *= vlen;
        int64_t src_offset = (blockIdx.z * r0 + index_selected) * s0 + idx_x;
        int64_t dst_offset =
            (blockIdx.z * num_indices + index_idx) * s0 + idx_x;

        int current_tail = 0;
        while (current_tail < tail) {
          out_ptr[dst_offset + current_tail] =
              in_ptr[src_offset + current_tail];
          ++current_tail;
        }
      }
    }
  }
}

template <typename SrcDtype, typename IndexDtype>
__global__ void IndexSelectKernel(
    SrcDtype* out_ptr,
    IndexDtype* index_ptr,
    SrcDtype* in_ptr,
    const int64_t r0,
    const int64_t s0,
    int64_t num_indices,
    const int64_t elements,
    const uint32_t tail) {
  (void)tail;
  int64_t index_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t index_selected = index_ptr[index_idx];
  int64_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;

  if (index_idx < num_indices && idx_x < s0) {
    int64_t src_offset = (blockIdx.z * r0 + index_selected) * s0 + idx_x;
    int64_t dst_offset = (blockIdx.z * num_indices + index_idx) * s0 + idx_x;
    out_ptr[dst_offset] = in_ptr[src_offset];
  }
}

#define GEN_FUNCTION(_INT, _IDXT)                                         \
  [](const Tensor& out,                                                   \
     const Tensor& index,                                                 \
     const Tensor& in,                                                    \
     const int64_t r0,                                                    \
     const int64_t s0,                                                    \
     const int64_t num_indices,                                           \
     dim3 block_size,                                                     \
     dim3 grid_size,                                                      \
     const int64_t elements,                                              \
     const uint32_t tail) {                                               \
    auto stream = c10::musa::getCurrentMUSAStream();                      \
    IndexSelectKernel<_INT, _IDXT><<<grid_size, block_size, 0, stream>>>( \
        static_cast<_INT*>(out.data_ptr()),                               \
        static_cast<_IDXT*>(index.data_ptr()),                            \
        static_cast<_INT*>(in.data_ptr()),                                \
        r0,                                                               \
        s0,                                                               \
        num_indices,                                                      \
        elements,                                                         \
        tail);                                                            \
  }

#define GEN_FUNCTION_VECTOR(_INT, _IDXT, _ALIGNED)   \
  [](const Tensor& out,                              \
     const Tensor& index,                            \
     const Tensor& in,                               \
     const int64_t r0,                               \
     const int64_t s0,                               \
     const int64_t num_indices,                      \
     dim3 block_size,                                \
     dim3 grid_size,                                 \
     const int64_t elements,                         \
     const uint32_t tail) {                          \
    auto stream = c10::musa::getCurrentMUSAStream(); \
    IndexSelectVectorKernel<_INT, _IDXT, _ALIGNED>   \
        <<<grid_size, block_size, 0, stream>>>(      \
            static_cast<_INT*>(out.data_ptr()),      \
            static_cast<_IDXT*>(index.data_ptr()),   \
            static_cast<_INT*>(in.data_ptr()),       \
            r0,                                      \
            s0,                                      \
            num_indices,                             \
            elements,                                \
            tail);                                   \
  }

#define REGISTER_KERNEL(_INT_ENUM, _CTYPE)                                 \
  index_select_vector_kernels[0][(int)_INT_ENUM][0] =                      \
      GEN_FUNCTION_VECTOR(_CTYPE, int32_t, false);                         \
  index_select_vector_kernels[1][(int)_INT_ENUM][0] =                      \
      GEN_FUNCTION_VECTOR(_CTYPE, int64_t, false);                         \
  index_select_vector_kernels[0][(int)_INT_ENUM][1] =                      \
      GEN_FUNCTION_VECTOR(_CTYPE, int32_t, true);                          \
  index_select_vector_kernels[1][(int)_INT_ENUM][1] =                      \
      GEN_FUNCTION_VECTOR(_CTYPE, int64_t, true);                          \
  index_select_kernels[0][(int)_INT_ENUM] = GEN_FUNCTION(_CTYPE, int32_t); \
  index_select_kernels[1][(int)_INT_ENUM] = GEN_FUNCTION(_CTYPE, int64_t);

struct KernelTable {
  using KernelFunc = std::function<void(
      const Tensor&,
      const Tensor&,
      const Tensor&,
      const int64_t,
      const int64_t,
      const int64_t,
      dim3,
      dim3,
      const int64_t,
      const uint32_t)>;

  KernelTable() {
    REGISTER_KERNEL(at::ScalarType::Half, float16_t);
    REGISTER_KERNEL(at::ScalarType::Float, float);
    REGISTER_KERNEL(at::ScalarType::Double, double);
    REGISTER_KERNEL(at::ScalarType::Int, int32_t);
    REGISTER_KERNEL(at::ScalarType::Long, int64_t);
    REGISTER_KERNEL(at::ScalarType::Char, int8_t);
    REGISTER_KERNEL(at::ScalarType::BFloat16, bfloat16_t);
    REGISTER_KERNEL(at::ScalarType::Byte, int8_t);
    REGISTER_KERNEL(at::ScalarType::Bool, int8_t);
    REGISTER_KERNEL(at::ScalarType::Short, int16_t);
    REGISTER_KERNEL(at::ScalarType::QInt8, int8_t);
    REGISTER_KERNEL(at::ScalarType::QUInt8, uint8_t);
  }

  void launch(
      const Tensor& out,
      const Tensor& index,
      const Tensor& in,
      const int64_t r0,
      const int64_t s0,
      const int64_t num_indices,
      dim3 block_size,
      dim3 grid_size,
      const int64_t elements,
      const uint32_t tail,
      bool can_vector_load) const {
    int index_dtype = index.scalar_type() == at::ScalarType::Int ? 0 : 1;
    auto& func = can_vector_load
        ? index_select_vector_kernels[index_dtype][(int)in.scalar_type()]
                                     [(int)(tail == 0)]
        : index_select_kernels[index_dtype][(int)in.scalar_type()];
    if (func) {
      func(
          out,
          index,
          in,
          r0,
          s0,
          num_indices,
          block_size,
          grid_size,
          elements,
          tail);
    } else {
      TORCH_CHECK(false, "IndexSelect unsupported!");
    }
  }

  static constexpr int index_types = 2; // int32 & int64
  static constexpr int nr_dtype = (int)at::ScalarType::NumOptions;
  static constexpr int nr_aligned = 2; // true & false
  KernelFunc index_select_vector_kernels[index_types][nr_dtype][nr_aligned];
  KernelFunc index_select_kernels[index_types][nr_dtype];
};
} // namespace

void IndexSelectRun(
    const int desc_dim,
    Tensor& out,
    const Tensor& index,
    const Tensor& in) {
  TORCH_CHECK(desc_dim < in.dim(), "Indexing dim is out of bounds");
  TORCH_CHECK(
      in.scalar_type() != at::ScalarType::QInt32,
      "Unsupported dtype of IndexSelect kernel input: ",
      out.scalar_type());

  int select_dim = desc_dim < 0 ? (desc_dim + in.dim()) : desc_dim;

  int64_t s1 = 1;
  for (int i = 0; i < select_dim; i++) {
    s1 *= in.size(i);
  }
  int64_t r0 = in.size(select_dim);
  int64_t s0 = 1;
  for (int j = select_dim + 1; j < in.dim(); j++) {
    s0 *= in.size(j);
  }
  int64_t elements = out.numel();
  int64_t num_indices = index.numel();
  bool can_vector_load = select_dim != in.dim() - 1;
  const int max_load_vlen = at::musa::DTypeSize(in.scalar_type()) <= 4
      ? (at::musa::DTypeSize(in.scalar_type()) <= 2 ? 8 : 4)
      : 2;
  const uint32_t tail = s0 % max_load_vlen;

  static KernelTable kernel_index_select;

  // thread blocks
  const int block_dim_x = 16;
  const int block_dim_y = 64;
  const uint32_t grid_dim_x = can_vector_load
      ? at::musa::ceil_div(
            at::musa::ceil_div(s0, (int64_t)max_load_vlen),
            (int64_t)block_dim_x)
      : at::musa::ceil_div(s0, (int64_t)block_dim_x);
  const uint32_t grid_dim_y =
      at::musa::ceil_div(num_indices, (int64_t)block_dim_y);
  const uint32_t grid_dim_z = s1;
  dim3 block_size{(uint32_t)block_dim_x, (uint32_t)block_dim_y, 1};
  dim3 grid_size{grid_dim_x, grid_dim_y, grid_dim_z};

  kernel_index_select.launch(
      out,
      index,
      in,
      r0,
      s0,
      num_indices,
      block_size,
      grid_size,
      elements,
      tail,
      can_vector_load);
}

REGISTER_MUSA_DISPATCH(indexselect_stub, &IndexSelectRun);

} // namespace native
} // namespace at
