#include <ATen/ATen.h>
#include <ATen/core/Array.h>
#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>

#include <musa_fp16.h>
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

typedef __half float16_t;

namespace {
constexpr int MAX_DIM = kMaxDim;

template <typename SrcDtype, typename IndexDtype, typename DstDtype = SrcDtype>
__global__ void IndexSelectVectorKernel(
    SrcDtype* out_ptr,
    IndexDtype* index_ptr,
    SrcDtype* in_ptr,
    const int s1,
    const int r0,
    const int s0,
    const int aligned_elements,
    int num_indices) {
  (void)s1;
  typedef typename at::musa::Dtype<SrcDtype>::Vec4 vec4;

  int index_idx = blockIdx.y * blockDim.y + threadIdx.y;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < aligned_elements;
       idx += blockDim.x * gridDim.x) {
    int index_selected = index_ptr[index_idx];
    int row = idx / (s0 >> 2);
    int col = (idx % (s0 >> 2)) << 2;
    int src_offset = (row * r0 + index_selected) * s0 + col;
    int dst_offset = (row * num_indices + index_idx) * s0 + col;
    *(vec4*)(out_ptr + dst_offset) = *(vec4*)(in_ptr + src_offset);
  }
}

template <typename SrcDtype, typename IndexDtype, typename DstDtype = SrcDtype>
__global__ void IndexSelectKernel(
    SrcDtype* out_ptr,
    IndexDtype* index_ptr,
    SrcDtype* in_ptr,
    const int s1,
    const int r0,
    const int s0,
    int num_indices) {
  int index_idx = blockIdx.y * blockDim.y + threadIdx.y;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < s1 * s0;
       idx += blockDim.x * gridDim.x) {
    int index_selected = index_ptr[index_idx];
    int row = idx / s0;
    int col = idx % s0;
    int src_offset = (row * r0 + index_selected) * s0 + col;
    int dst_offset = (row * num_indices + index_idx) * s0 + col;
    out_ptr[dst_offset] = in_ptr[src_offset];
  }
}

#define GEN_FUNCTION(_INT, _IDXT)                                         \
  [](const Tensor& out,                                                   \
     const Tensor& index,                                                 \
     const Tensor& in,                                                    \
     const int s1,                                                        \
     const int r0,                                                        \
     const int s0,                                                        \
     const int aligned_elements,                                          \
     const int num_indices,                                               \
     const uint32_t nr_block,                                             \
     const uint32_t nr_threads) {                                         \
    (void)aligned_elements;                                               \
    auto stream = c10::musa::getCurrentMUSAStream();                      \
    dim3 grid_size{nr_block, static_cast<uint32_t>(num_indices), 1};      \
    IndexSelectKernel<_INT, _IDXT><<<grid_size, nr_threads, 0, stream>>>( \
        static_cast<_INT*>(out.data_ptr()),                               \
        static_cast<_IDXT*>(index.data_ptr()),                            \
        static_cast<_INT*>(in.data_ptr()),                                \
        s1,                                                               \
        r0,                                                               \
        s0,                                                               \
        num_indices);                                                     \
  }

#define GEN_FUNCTION_VECTOR(_INT, _IDXT)                             \
  [](const Tensor& out,                                              \
     const Tensor& index,                                            \
     const Tensor& in,                                               \
     const int s1,                                                   \
     const int r0,                                                   \
     const int s0,                                                   \
     const int aligned_elements,                                     \
     const int num_indices,                                          \
     const uint32_t nr_block,                                        \
     const uint32_t nr_threads) {                                    \
    auto stream = c10::musa::getCurrentMUSAStream();                 \
    dim3 grid_size{nr_block, static_cast<uint32_t>(num_indices), 1}; \
    IndexSelectVectorKernel<_INT, _IDXT>                             \
        <<<grid_size, nr_threads, 0, stream>>>(                      \
            static_cast<_INT*>(out.data_ptr()),                      \
            static_cast<_IDXT*>(index.data_ptr()),                   \
            static_cast<_INT*>(in.data_ptr()),                       \
            s1,                                                      \
            r0,                                                      \
            s0,                                                      \
            aligned_elements,                                        \
            num_indices);                                            \
  }

#define REGISTER_KERNEL(_INT_ENUM, _CTYPE)                                 \
  index_select_vector_kernels[0][(int)_INT_ENUM] =                         \
      GEN_FUNCTION_VECTOR(_CTYPE, int32_t);                                \
  index_select_vector_kernels[1][(int)_INT_ENUM] =                         \
      GEN_FUNCTION_VECTOR(_CTYPE, int64_t);                                \
  index_select_kernels[0][(int)_INT_ENUM] = GEN_FUNCTION(_CTYPE, int32_t); \
  index_select_kernels[1][(int)_INT_ENUM] = GEN_FUNCTION(_CTYPE, int64_t);

struct KernelTable {
  using KernelFunc = std::function<void(
      const Tensor&,
      const Tensor&,
      const Tensor&,
      const int,
      const int,
      const int,
      const int,
      const int,
      const int,
      const int)>;

  KernelTable() {
    const int index_tyeps = 2; // int32 & int64
    const int nr_dtype = (int)at::ScalarType::NumOptions;
    index_select_kernels.resize(index_tyeps, std::vector<KernelFunc>(nr_dtype));
    index_select_vector_kernels.resize(
        index_tyeps, std::vector<KernelFunc>(nr_dtype));

    REGISTER_KERNEL(at::ScalarType::Half, float16_t);
    REGISTER_KERNEL(at::ScalarType::Float, float);
    REGISTER_KERNEL(at::ScalarType::Double, double);
    REGISTER_KERNEL(at::ScalarType::Int, int32_t);
    REGISTER_KERNEL(at::ScalarType::Long, int64_t);
    REGISTER_KERNEL(at::ScalarType::Char, int8_t);
  }

  void launch(
      const Tensor& out,
      const Tensor& index,
      const Tensor& in,
      const int s1,
      const int r0,
      const int s0,
      const int aligned_elements,
      const int num_indices,
      const uint32_t nr_block,
      const uint32_t nr_threads,
      bool can_vector_load) const {
    int index_dtype = index.scalar_type() == at::ScalarType::Int ? 0 : 1;
    auto& func = can_vector_load
        ? index_select_vector_kernels[index_dtype][(int)in.scalar_type()]
        : index_select_kernels[index_dtype][(int)in.scalar_type()];
    if (func) {
      func(
          out,
          index,
          in,
          s1,
          r0,
          s0,
          aligned_elements,
          num_indices,
          nr_block,
          nr_threads);
    } else {
      TORCH_CHECK(false, "IndexSelect unsupported!");
    }
  }

  std::vector<std::vector<KernelFunc>> index_select_vector_kernels;
  std::vector<std::vector<KernelFunc>> index_select_kernels;
};
} // namespace

void IndexSelectRun(
    const int desc_dim,
    Tensor& out,
    const Tensor& index,
    const Tensor& in) {
  TORCH_CHECK(desc_dim < in.dim(), "Indexing dim is out of bounds");
  TORCH_CHECK(
      (in.scalar_type() == at::ScalarType::Float) ||
          (in.scalar_type() == at::ScalarType::Long) ||
          (in.scalar_type() == at::ScalarType::Int) ||
          (in.scalar_type() == at::ScalarType::Char) ||
          (in.scalar_type() == at::ScalarType::Half) ||
          (in.scalar_type() == at::ScalarType::Double),
      "Index only support input dtype float16/32/64, int32/64, but got ",
      out.scalar_type());

  int select_dim = desc_dim < 0 ? (desc_dim + in.dim()) : desc_dim;

  int s1 = 1;
  for (int i = 0; i < select_dim; i++) {
    s1 *= in.sizes()[i];
  }
  int r0 = in.sizes()[select_dim];
  int s0 = 1;
  for (int j = select_dim + 1; j < in.dim(); j++) {
    s0 *= in.sizes()[j];
  }
  int num_indices = index.numel();

  bool can_vector_load = (select_dim != in.dim() - 1) && (s0 % 4 == 0);

  // device info
  musaDeviceProp device_prop;
  at::musa::muHandle& h = GetMudnnHandle();
  int device_id = h.GetDeviceId();
  TORCH_CHECK(
      musaSuccess == musaGetDeviceProperties(&device_prop, device_id),
      "musaGetDeviceProperties error");
  int max_block_num = device_prop.multiProcessorCount;

  static KernelTable kernel_index_select;
  const uint32_t nr_threads = 512;
  const uint32_t nr_blocks = can_vector_load
      ? std::min(at::musa::ceil_div(s0 / 4 * s1, 512), max_block_num)
      : std::min(at::musa::ceil_div(s0 * s1, 512), max_block_num);
  kernel_index_select.launch(
      out,
      index,
      in,
      s1,
      r0,
      s0,
      s0 / 4 * s1,
      num_indices,
      nr_blocks,
      nr_threads,
      can_vector_load);
}

REGISTER_MUSA_DISPATCH(indexselect_stub, &IndexSelectRun);

} // namespace native
} // namespace at
