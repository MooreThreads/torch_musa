#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/ops/Triangular.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace native {

namespace {

template <typename T, TriangularMode mode>
__global__ void triangular_matrix(
    T* out_ptr,
    const T* in_ptr,
    const int M,
    const int N,
    const int64_t MxN,
    const int diag,
    const int64_t elements) {
  typedef typename at::musa::Dtype<T>::Vec4 vec4;

  int gid = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
  int global_stride = 4 * blockDim.x * gridDim.x;
  int num_matrix = gid / MxN + 1;
  int prefix = MxN * num_matrix;

  if (gid < elements) {
    vec4 vec_out, vec_in;
    for (int offset = gid; offset < elements; offset += global_stride) {
      num_matrix = offset / MxN + 1;
      prefix = MxN * num_matrix;
      if (offset + 4 < elements) {
        vec_in = *(const vec4*)(in_ptr + offset);
        int carry = 1;
        bool mask[4] = {true, true, true, true};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (offset + i >= prefix) {
            carry = 0;
          }
          int col = (offset - MxN * (num_matrix - carry) + i) % N;
          int row = (offset - MxN * (num_matrix - carry) + i) / N;

          if constexpr (mode == TriangularMode::TRIU) {
            mask[i] = ((col) - (row) >= (diag));
          } else {
            mask[i] = ((col) - (row) <= (diag));
          }
        }
        vec_out.x = mask[0] ? vec_in.x : (T)0;
        vec_out.y = mask[1] ? vec_in.y : (T)0;
        vec_out.z = mask[2] ? vec_in.z : (T)0;
        vec_out.w = mask[3] ? vec_in.w : (T)0;

        *(vec4*)(out_ptr + offset) = vec_out;
      } else {
        for (int offset_dst = offset; offset_dst < elements; offset_dst++) {
          int col = (offset_dst - MxN * (num_matrix - 1)) % N;
          int row = (offset_dst - MxN * (num_matrix - 1)) / N;

          bool mask = true;
          if constexpr (mode == TriangularMode::TRIU) {
            mask = ((col) - (row) >= (diag));
          } else {
            mask = ((col) - (row) <= (diag));
          }
          out_ptr[offset_dst] = mask ? in_ptr[offset_dst] : (T)0;
        }
      }
    }
  }
}

template <typename T, TriangularMode mode>
__global__ void triangular_matrix_slow(
    T* out_ptr,
    const T* in_ptr,
    const int M,
    const int N,
    const int64_t MxN,
    const int diag,
    const int64_t elements) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int global_stride = blockDim.x * gridDim.x;
  int num_matrix = gid / MxN + 1;

  if (gid < elements) {
    for (int offset = gid; offset < elements; offset += global_stride) {
      num_matrix = offset / MxN + 1;
      int col = (offset - MxN * (num_matrix - 1)) % N;
      int row = (offset - MxN * (num_matrix - 1)) / N;
      bool mask = true;
      if constexpr (mode == TriangularMode::TRIU) {
        mask = ((col) - (row) >= (diag));
      } else {
        mask = ((col) - (row) <= (diag));
      }
      out_ptr[offset] = mask ? in_ptr[offset] : (T)0;
    }
  }
}

#define GEN_TRI_MAT_FUNC(_MODE, _TYPE)                                       \
  [](Tensor& out,                                                            \
     const Tensor& in,                                                       \
     const int M,                                                            \
     const int N,                                                            \
     const int64_t MxN,                                                      \
     const int diag,                                                         \
     const int block_num,                                                    \
     const int block_size,                                                   \
     const int64_t elements) {                                               \
    auto stream = c10::musa::getCurrentMUSAStream();                         \
    if (MxN >= 4) {                                                          \
      triangular_matrix<_TYPE, _MODE><<<block_num, block_size, 0, stream>>>( \
          static_cast<_TYPE*>(out.data_ptr()),                               \
          static_cast<_TYPE*>(in.data_ptr()),                                \
          M,                                                                 \
          N,                                                                 \
          MxN,                                                               \
          diag,                                                              \
          elements);                                                         \
    } else {                                                                 \
      triangular_matrix_slow<_TYPE, _MODE>                                   \
          <<<block_num, block_size, 0, stream>>>(                            \
              static_cast<_TYPE*>(out.data_ptr()),                           \
              static_cast<_TYPE*>(in.data_ptr()),                            \
              M,                                                             \
              N,                                                             \
              MxN,                                                           \
              diag,                                                          \
              elements);                                                     \
    }                                                                        \
  }
} // namespace

#define REGISTER_KERNEL(_TYPE, _MODE, _CTYPE) \
  triangular_kernels[(int)_TYPE][(int)_MODE] = GEN_TRI_MAT_FUNC(_MODE, _CTYPE);

struct KernelTable {
  using KernelFunc = std::function<void(
      Tensor&,
      const Tensor&,
      const int,
      const int,
      const int64_t,
      const int,
      const int,
      const int,
      const int64_t)>;

#define REGISTER_KERNEL_MODE(_MODE)                            \
  REGISTER_KERNEL(at::ScalarType::Bool, _MODE, bool)           \
  REGISTER_KERNEL(at::ScalarType::Byte, _MODE, uint8_t)        \
  REGISTER_KERNEL(at::ScalarType::Half, _MODE, float16_t)      \
  REGISTER_KERNEL(at::ScalarType::BFloat16, _MODE, bfloat16_t) \
  REGISTER_KERNEL(at::ScalarType::Float, _MODE, float)         \
  REGISTER_KERNEL(at::ScalarType::Double, _MODE, double)       \
  REGISTER_KERNEL(at::ScalarType::Char, _MODE, int8_t)         \
  REGISTER_KERNEL(at::ScalarType::Int, _MODE, int32_t)         \
  REGISTER_KERNEL(at::ScalarType::Long, _MODE, int64_t)

  KernelTable() {
    REGISTER_KERNEL_MODE(TriangularMode::TRIU);
    REGISTER_KERNEL_MODE(TriangularMode::TRIL);
  }
#undef REGISTER_KERNEL_MODE

  template <TriangularMode mode>
  void launch(
      Tensor& out,
      const Tensor& in,
      const int M,
      const int N,
      const int64_t MxN,
      const int diag,
      const int block_num,
      const int block_size,
      const int64_t elements) {
    at::ScalarType dtype = in.scalar_type();
    auto& func = triangular_kernels[(int)dtype][(int)mode];

    if (func) {
      func(out, in, M, N, MxN, diag, block_num, block_size, elements);
    } else {
      TORCH_CHECK(false, "Unsupported dtype of Triangular: ", dtype);
    }
  }

  static constexpr int nr_mode = 2; // triu && tril
  static constexpr int nr_dtype = (int)at::ScalarType::NumOptions;
  KernelFunc triangular_kernels[nr_dtype][nr_mode];
};

bool CheckParams(const Tensor& o, const Tensor& i) {
  if (o.dim() != i.dim() && o.dim() >= 2) {
    return false;
  }
  if (i.scalar_type() != o.scalar_type()) {
    return false;
  }

  for (int j = 0; j < 2; ++j) {
    if (o.sizes()[j] != i.sizes()[j]) {
      return false;
    }
  }
  return true;
}

void TriuRun(Tensor& o, const Tensor& i, const int64_t diag) {
  TORCH_CHECK(CheckParams(o, i), "CheckParams fail");
  at::musa::muHandle& h = GetMudnnHandle();
  const int ndim = o.dim();
  const int elements = o.numel();
  const int M = o.size(ndim - 2);
  const int N = o.size(ndim - 1);
  const int MxN = M * N;

  // device info
  musaDeviceProp device_prop;
  int device_id = h.GetDeviceId();
  TORCH_CHECK(
      musaSuccess == musaGetDeviceProperties(&device_prop, device_id),
      "musaGetDeviceProperties error");
  const int mp_num = device_prop.multiProcessorCount;
  const int block_size = 1024;
  const int max_blocks = INT32_MAX;
  const int block_num =
      std::min(at::musa::ceil_div(elements, block_size * 4), max_blocks);

  KernelTable kernel_triangular;
  kernel_triangular.launch<TriangularMode::TRIU>(
      o, i, M, N, MxN, static_cast<int>(diag), block_num, block_size, elements);
}

void TrilRun(Tensor& o, const Tensor& i, const int64_t diag) {
  TORCH_CHECK(CheckParams(o, i), "CheckParams fail");
  at::musa::muHandle& h = GetMudnnHandle();
  const int ndim = o.dim();
  const int elements = o.numel();
  const int M = o.size(ndim - 2);
  const int N = o.size(ndim - 1);
  const int MxN = M * N;

  // device info
  musaDeviceProp device_prop;
  int device_id = h.GetDeviceId();
  TORCH_CHECK(
      musaSuccess == musaGetDeviceProperties(&device_prop, device_id),
      "musaGetDeviceProperties error");
  const int mp_num = device_prop.multiProcessorCount;
  const int block_size = 1024;
  const int max_blocks = INT32_MAX;
  const int block_num =
      std::min(at::musa::ceil_div(elements, block_size * 4), max_blocks);

  KernelTable kernel_triangular;
  kernel_triangular.launch<TriangularMode::TRIL>(
      o, i, M, N, MxN, static_cast<int>(diag), block_num, block_size, elements);
}

REGISTER_MUSA_DISPATCH(triu_stub, &TriuRun);
REGISTER_MUSA_DISPATCH(tril_stub, &TrilRun);

} // namespace native
} // namespace at
