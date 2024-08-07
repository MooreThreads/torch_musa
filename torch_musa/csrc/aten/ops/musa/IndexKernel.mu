#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Array.h>
#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/IndexKernel.h>

#include "torch_musa/csrc/aten/musa/MUSAAtomic.muh"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/ops/TensorShape.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <functional>
#include <tuple>

namespace at {
namespace native {

using at::detail::Array;
using at::musa::ceil_div;
using at::musa::FastDivmod;
using at::musa::VecType;

namespace {
constexpr int MAX_DIM = 8;
constexpr int NARGS = 3; // output, input, indices

struct IndexKernelLaunchInfo {
  bool can_vector_load{false};
  bool no_divider{false};
  int32_t ndim{0};
};

inline void ConfigIndexKernelLaunchInfo(
    TensorIteratorBase& iter,
    IndexKernelLaunchInfo& info) {
  int ndim = iter.ndim();
  int element_size = iter.element_size(0);
  const int64_t* out_strides_ptr = iter.strides(0).data();
  const int64_t* in_strides_ptr = iter.strides(1).data();

  info.can_vector_load =
      (out_strides_ptr[0] == in_strides_ptr[0] &&
       (out_strides_ptr[0] / element_size) == 1);
  info.ndim = ndim;
  info.no_divider = (ndim <= 3);
}

template <typename func_t>
__global__ void IndexElementwiseKernel(
    int64_t x_max,
    int64_t y_max,
    int64_t z_max,
    func_t f) {
  uint32_t x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t z_idx = blockIdx.z * blockDim.z + threadIdx.z;

  if ((x_idx < x_max) && y_idx < y_max && z_idx < z_max) {
    f(x_idx, y_idx, z_idx);
  }
}

template <typename vec_func_t, typename ew_func_t>
__global__ void IndexVectorLoadKernel(
    int64_t x_max,
    int64_t y_max,
    int64_t z_max,
    int vlen,
    vec_func_t vec_func,
    ew_func_t ew_func) {
  uint32_t x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t z_idx = blockIdx.z * blockDim.z + threadIdx.z;

  uint32_t contig_idx = x_idx * vlen;
  if (((contig_idx + vlen) <= x_max) && y_idx < y_max && z_idx < z_max) {
    vec_func(contig_idx, y_idx, z_idx);
    contig_idx += (gridDim.x * blockDim.x * vlen);
  }

  while (y_idx < y_max && z_idx < z_max && contig_idx < x_max) {
    ew_func(contig_idx, y_idx, z_idx);
    contig_idx++;
  }
}

template <typename func_t>
void LaunchElementwiseKernel(
    dim3 grid,
    dim3 block,
    int64_t x_max,
    int64_t y_max,
    int64_t z_max,
    const func_t& f) {
  auto stream = at::musa::getCurrentMUSAStream();
  IndexElementwiseKernel<<<grid, block, 0, stream>>>(x_max, y_max, z_max, f);
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

template <typename vec_func_t, typename ew_func_t>
void LaunchVectorLoadKernel(
    dim3 grid,
    dim3 block,
    int64_t x_max,
    int64_t y_max,
    int64_t z_max,
    int vlen,
    const vec_func_t& f1,
    const ew_func_t& f2) {
  auto stream = at::musa::getCurrentMUSAStream();
  IndexVectorLoadKernel<<<grid, block, 0, stream>>>(
      x_max, y_max, z_max, vlen, f1, f2);
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

template <
    const int MAX_THREE = 3,
    const bool VEC_LOAD = false,
    const bool NO_DIVIDER = false,
    const int VLEN = 1,
    typename... NFuncs>
void GPUIndexKernel(
    TensorIteratorBase& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    NFuncs... funcs) {
  auto device_funcs = std::make_tuple(funcs...);
  int num_indices = index_size.size();

  if (iter.numel() == 0) {
    return;
  }
  const int ndim = iter.ndim();

  auto sizes = Array<int64_t, MAX_DIM>(1);
  auto strides = Array<int64_t, NARGS * MAX_DIM>(0);
  const int64_t* sizes_ptr = iter.shape().data();
  for (int i = 0; i < ndim; i++) {
    sizes[i] = sizes_ptr[i];
  }
  for (int i = 0; i < NARGS; i++) {
    const int64_t* strides_ptr = iter.strides(i).data();
    for (int j = 0; j < ndim; j++) {
      strides[i * MAX_DIM + j] = strides_ptr[j];
    }
  }
  auto indexed_strides = Array<int64_t, MAX_DIM>(0);
  auto index_ptrs = Array<char*, MAX_DIM>(nullptr);
  for (int i = 0; i < num_indices; i++) {
    // index_stride is in # of bytes
    indexed_strides[i] = index_stride[i];
    index_ptrs[i] = (char*)iter.data_ptr(i + 2);
  }

  dim3 block_size = {1, 1, 1};
  block_size.x = 256;
  int num_thread_x = ceil_div(int(sizes[0]), VLEN);
  if (num_thread_x <= 16) {
    block_size.x = 4;
  } else if (num_thread_x <= 64) {
    block_size.x = 16;
  } else if (num_thread_x <= 256) {
    block_size.x = 64;
  } else if (num_thread_x <= 512) {
    block_size.x = 128;
  }
  block_size.y = 1024 / block_size.x;

  const uint32_t grid_dim_x = ceil_div((int)sizes[0], block_size.x * VLEN);
  const uint32_t grid_dim_y = ceil_div((int)sizes[1], (int)block_size.y);
  uint32_t grid_dim_z = 1;
  for (int i = 2; i < ndim; i++) {
    grid_dim_z *= sizes[i];
  }
  dim3 grid_size = {grid_dim_x, grid_dim_y, grid_dim_z};

  char* out_data = (char*)iter.data_ptr(0);
  char* in_data = (char*)iter.data_ptr(1);

  if constexpr (NO_DIVIDER) {
#define CALCULATE_OFFSETS_WITHOUT_DIVIDER(x_idx, y_idx, z_idx) \
  int64_t offsets[NARGS] = {0};                                \
  int64_t idxs[3] = {x_idx, y_idx, z_idx};                     \
  _Pragma("unroll") for (int i = 0; i < NARGS; i++) {          \
    _Pragma("unroll") for (int j = 0; j < MAX_THREE; j++) {    \
      offsets[i] += idxs[j] * strides[i * MAX_DIM + j];        \
    }                                                          \
  }                                                            \
  int64_t offset = 0;                                          \
  for (int i = 0; i < num_indices; i++) {                      \
    int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);   \
    offset += index * indexed_strides[i];                      \
  }                                                            \
  char* cur_out_ptr = out_data + offsets[0];                   \
  char* cur_in_ptr = in_data + offsets[1];

    if constexpr (VEC_LOAD) {
      static_assert(
          sizeof...(NFuncs) == 2, "Invalid number of function argument");
      LaunchVectorLoadKernel(
          grid_size,
          block_size,
          sizes[0],
          sizes[1],
          sizes[2],
          VLEN,
          [=] __device__(uint32_t x_idx, uint32_t y_idx, uint32_t z_idx) {
            CALCULATE_OFFSETS_WITHOUT_DIVIDER(x_idx, y_idx, z_idx);
            std::get<1>(device_funcs)(cur_out_ptr, cur_in_ptr, offset);
          },
          [=] __device__(uint32_t x_idx, uint32_t y_idx, uint32_t z_idx) {
            CALCULATE_OFFSETS_WITHOUT_DIVIDER(x_idx, y_idx, z_idx);
            std::get<0>(device_funcs)(cur_out_ptr, cur_in_ptr, offset);
          });

    } else {
      LaunchElementwiseKernel(
          grid_size,
          block_size,
          sizes[0],
          sizes[1],
          sizes[2],
          [=] __device__(uint32_t x_idx, uint32_t y_idx, uint32_t z_idx) {
            CALCULATE_OFFSETS_WITHOUT_DIVIDER(x_idx, y_idx, z_idx);
            std::get<0>(device_funcs)(cur_out_ptr, cur_in_ptr, offset);
          });
    }

#undef CALCULATE_OFFSETS_WITHOUT_DIVIDER
  } else {
    // seems we have no choice but to use divider in this case (ndim > 3)
    Array<FastDivmod, MAX_DIM> z_sizes_fastdv;
    for (int i = 2; i < ndim; i++) {
      z_sizes_fastdv[i] = FastDivmod((uint32_t)sizes[i]);
    }

#define CALCULATE_OFFSETS_WITH_DIVIDER(x_idx, y_idx, z_idx)   \
  int64_t offsets[NARGS] = {0};                               \
  int64_t idxs[2] = {x_idx, y_idx};                           \
  _Pragma("unroll") for (int i = 0; i < NARGS; i++) {         \
    _Pragma("unroll") for (int j = 0; j < 2; j++) {           \
      offsets[i] += idxs[j] * strides[i * MAX_DIM + j];       \
    }                                                         \
  }                                                           \
  _Pragma("unroll") for (int dim = 2; dim < MAX_DIM; dim++) { \
    if (dim == ndim) {                                        \
      break;                                                  \
    }                                                         \
    uint32_t q, index;                                        \
    z_sizes_fastdv[dim](q, index, z_idx);                     \
    z_idx = q;                                                \
    _Pragma("unroll") for (int n = 0; n < NARGS; n++) {       \
      offsets[n] += index * strides[n * MAX_DIM + dim];       \
    }                                                         \
  }                                                           \
  int64_t offset = 0;                                         \
  for (int i = 0; i < num_indices; i++) {                     \
    int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);  \
    offset += index * indexed_strides[i];                     \
  }                                                           \
  char* cur_out_ptr = out_data + offsets[0];                  \
  char* cur_in_ptr = in_data + offsets[1];

    if constexpr (VEC_LOAD) {
      static_assert(
          sizeof...(NFuncs) == 2, "Invalid number of function argument");
      LaunchVectorLoadKernel(
          grid_size,
          block_size,
          sizes[0],
          sizes[1],
          grid_dim_z,
          VLEN,
          [=] __device__(uint32_t x_idx, uint32_t y_idx, uint32_t z_idx) {
            CALCULATE_OFFSETS_WITH_DIVIDER(x_idx, y_idx, z_idx);
            std::get<1>(device_funcs)(cur_out_ptr, cur_in_ptr, offset);
          },
          [=] __device__(uint32_t x_idx, uint32_t y_idx, uint32_t z_idx) {
            CALCULATE_OFFSETS_WITH_DIVIDER(x_idx, y_idx, z_idx);
            std::get<0>(device_funcs)(cur_out_ptr, cur_in_ptr, offset);
          });

    } else {
      LaunchElementwiseKernel(
          grid_size,
          block_size,
          sizes[0],
          sizes[1],
          grid_dim_z,
          [=] __device__(uint32_t x_idx, uint32_t y_idx, uint32_t z_idx) {
            CALCULATE_OFFSETS_WITH_DIVIDER(x_idx, y_idx, z_idx);
            std::get<0>(device_funcs)(cur_out_ptr, cur_in_ptr, offset);
          });
    }
#undef CALCULATE_OFFSETS_WITH_DIVIDER
  }
}

template <typename scalar_t>
void IndexPutKernelImpl(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride) {
  constexpr int64_t vlen = sizeof(scalar_t) <= 4
      ? (sizeof(scalar_t) <= 2 ? (sizeof(scalar_t) <= 1 ? 16 : 8) : 4)
      : 2;
  using vec_dtype = VecType<scalar_t, vlen * sizeof(scalar_t) * 8>;
  IndexKernelLaunchInfo info;
  ConfigIndexKernelLaunchInfo(iter, info);

#define DISPATCH_VECTOR_KERNEL(max_three, no_divider, vlen)             \
  GPUIndexKernel<max_three, true, no_divider, vlen>(                    \
      iter,                                                             \
      index_size,                                                       \
      index_stride,                                                     \
      [] __device__(char* out_data, char* in_data, int64_t offset) {    \
        *(scalar_t*)(out_data + offset) = *(scalar_t*)in_data;          \
      },                                                                \
      [] __device__(char* out_data, char* in_data, int64_t offset) {    \
        vec_dtype value_reg = vec_dtype::load((scalar_t*)in_data, 0);   \
        vec_dtype::store((scalar_t*)(out_data + offset), 0, value_reg); \
      });

#define DISPATCH_KERNEL(max_three, no_divider)                       \
  GPUIndexKernel<max_three, false, no_divider, 1>(                   \
      iter,                                                          \
      index_size,                                                    \
      index_stride,                                                  \
      [] __device__(char* out_data, char* in_data, int64_t offset) { \
        *(scalar_t*)(out_data + offset) = *(scalar_t*)in_data;       \
      });

  if (info.can_vector_load) {
    if (info.no_divider) {
      if (info.ndim == 1) {
        DISPATCH_VECTOR_KERNEL(1, true, vlen);
      } else if (info.ndim == 2) {
        DISPATCH_VECTOR_KERNEL(2, true, vlen);
      } else {
        DISPATCH_VECTOR_KERNEL(3, true, vlen);
      }
    } else {
      DISPATCH_VECTOR_KERNEL(3, false, vlen);
    }
  } else {
    if (info.no_divider) {
      if (info.ndim == 1) {
        DISPATCH_KERNEL(1, true);
      } else if (info.ndim == 2) {
        DISPATCH_KERNEL(2, true);
      } else {
        DISPATCH_KERNEL(3, true);
      }
    } else {
      DISPATCH_KERNEL(3, false);
    }
  }

#undef DISPATCH_KERNEL
#undef DISPATCH_VECTOR_KERNEL
}

template <typename scalar_t>
void IndexPutAtomicAddKernelImpl(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride) {
  IndexKernelLaunchInfo info;
  ConfigIndexKernelLaunchInfo(iter, info);
  info.can_vector_load = false;
#define DISPATCH_ATOMIC_KERNEL(max_three, no_divider)                \
  GPUIndexKernel<max_three, false, no_divider, 1>(                   \
      iter,                                                          \
      index_size,                                                    \
      index_stride,                                                  \
      [] __device__(char* out_data, char* in_data, int64_t offset) { \
        at::musa::gpuAtomicAdd(                                      \
            (scalar_t*)(out_data + offset), *(scalar_t*)in_data);    \
      });

  if (info.no_divider) {
    if (info.ndim == 1) {
      DISPATCH_ATOMIC_KERNEL(1, true);
    } else if (info.ndim == 2) {
      DISPATCH_ATOMIC_KERNEL(2, true);
    } else {
      DISPATCH_ATOMIC_KERNEL(3, true);
    }
  } else {
    DISPATCH_ATOMIC_KERNEL(3, false);
  }

#undef DISPATCH_ATOMIC_KERNEL
}

template <typename scalar_t>
void IndexKernelImpl(
    TensorIteratorBase& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride) {
  constexpr int64_t vlen = sizeof(scalar_t) <= 4
      ? (sizeof(scalar_t) <= 2 ? (sizeof(scalar_t) <= 1 ? 16 : 8) : 4)
      : 2;
  using vec_dtype = VecType<scalar_t, vlen * sizeof(scalar_t) * 8>;
  IndexKernelLaunchInfo info;
  ConfigIndexKernelLaunchInfo(iter, info);

#define DISPATCH_VECTOR_KERNEL(max_three, no_divider, vlen)          \
  GPUIndexKernel<max_three, true, no_divider, vlen>(                 \
      iter,                                                          \
      index_size,                                                    \
      index_stride,                                                  \
      [] __device__(char* out_data, char* in_data, int64_t offset) { \
        *(scalar_t*)out_data = *(scalar_t*)(in_data + offset);       \
      },                                                             \
      [] __device__(char* out_data, char* in_data, int64_t offset) { \
        vec_dtype value_reg =                                        \
            vec_dtype::load((scalar_t*)(in_data + offset), 0);       \
        vec_dtype::store((scalar_t*)out_data, 0, value_reg);         \
      });

#define DISPATCH_KERNEL(max_three, no_divider)                       \
  GPUIndexKernel<max_three, false, no_divider, 1>(                   \
      iter,                                                          \
      index_size,                                                    \
      index_stride,                                                  \
      [] __device__(char* out_data, char* in_data, int64_t offset) { \
        *(scalar_t*)out_data = *(scalar_t*)(in_data + offset);       \
      });

  if (info.can_vector_load) {
    if (info.no_divider) {
      if (info.ndim == 1) {
        DISPATCH_VECTOR_KERNEL(1, true, vlen);
      } else if (info.ndim == 2) {
        DISPATCH_VECTOR_KERNEL(2, true, vlen);
      } else {
        DISPATCH_VECTOR_KERNEL(3, true, vlen);
      }
    } else {
      DISPATCH_VECTOR_KERNEL(3, false, vlen);
    }
  } else {
    if (info.no_divider) {
      if (info.ndim == 1) {
        DISPATCH_KERNEL(1, true);
      } else if (info.ndim == 2) {
        DISPATCH_KERNEL(2, true);
      } else {
        DISPATCH_KERNEL(3, true);
      }
    } else {
      DISPATCH_KERNEL(3, false);
    }
  }
}

#undef DISPATCH_KERNEL
#undef DISPATCH_VECTOR_KERNEL

} // anonymous namespace

static void IndexPutKernel(
    TensorIterator& iter,
    IntArrayRef indexed_size,
    IntArrayRef indexed_stride,
    bool accumulate) {
  if (accumulate) {
    AT_DISPATCH_ALL_TYPES_AND3(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        at::ScalarType::Bool,
        iter.dtype(),
        "IndexPut",
        [&] {
          IndexPutAtomicAddKernelImpl<scalar_t>(
              iter, indexed_size, indexed_stride);
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        at::ScalarType::Bool,
        iter.dtype(),
        "IndexPut",
        [&] {
          IndexPutKernelImpl<scalar_t>(iter, indexed_size, indexed_stride);
        });
  }
}

static void IndexKernel(
    TensorIteratorBase& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride) {
  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.dtype(),
      "IndexMusa",
      [&] { IndexKernelImpl<scalar_t>(iter, index_size, index_stride); });
}

REGISTER_MUSA_DISPATCH(index_put_stub, &IndexPutKernel);
REGISTER_MUSA_DISPATCH(index_stub, &IndexKernel);

} // namespace native
} // namespace at
