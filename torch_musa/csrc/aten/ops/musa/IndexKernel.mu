#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Array.h>
#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/musa/detail/IndexUtils.muh>
#include <ATen/musa/detail/OffsetCalculator.muh>
#include <ATen/native/musa/KernelUtils.muh>

#include "torch_musa/csrc/aten/musa/MUSAAtomic.muh"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/core/MUSAStream.h"

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
static constexpr int LAUNCH_BOUND2 = 4;
static constexpr int LAUNCH_SIZE_ND = 128;

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

template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, LAUNCH_BOUND2)
__global__ void IndexElementwiseKernel(const int64_t N, const func_t f) {
  const auto tid = threadIdx.x;
  const auto nv = nt * vt;
  auto idx = nv * blockIdx.x + tid;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
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

template <int nt, int vt, typename func_t>
static void LaunchKernel(const int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  const auto stream = at::musa::getCurrentMUSAStream();
  IndexElementwiseKernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  C10_MUSA_KERNEL_LAUNCH_CHECK();
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

  auto sizes = Array<int64_t, MAX_DIM>(1); // target_size
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
  auto indexed_size = Array<int64_t, MAX_DIM>(0);
  auto indexed_strides = Array<int64_t, MAX_DIM>(0);
  auto index_ptrs = Array<char*, MAX_DIM>(nullptr);
  for (int i = 0; i < num_indices; i++) {
    // index_stride is in # of bytes
    indexed_strides[i] = index_stride[i];
    index_ptrs[i] = (char*)iter.data_ptr(i + 2);
    indexed_size[i] = index_size[i];
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

  const uint32_t grid_dim_x = ceil_div(sizes[0], (int64_t)block_size.x * VLEN);
  const uint32_t grid_dim_y = ceil_div(sizes[1], (int64_t)block_size.y);
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
    if (index < 0) {                                           \
      index += indexed_size[i];                                \
    }                                                          \
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
    if (index < 0) {                                          \
      index += indexed_size[i];                               \
    }                                                         \
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

template <typename scalar_t, typename index_t, typename func_t>
void MusaTakePutKernel(
    TensorIterator& iter,
    const TensorBase& indexed,
    const func_t& f) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      MusaTakePutKernel<scalar_t, index_t>(sub_iter, indexed, f);
    }
    return;
  }

  const auto numel = indexed.numel();
  const bool is_contiguous = indexed.is_contiguous();

  char* const __restrict__ iterated_ptr =
      reinterpret_cast<char*>(iter.data_ptr(0));
  char* const __restrict__ idx_ptr = reinterpret_cast<char*>(iter.data_ptr(1));

  const auto offset_calc = make_offset_calculator<2>(iter);
  using uindex_t = std::make_unsigned_t<index_t>;

  // OffsetCalculator needs the sizes and strides reveresed
  const auto indexed_sizes =
      std::vector<int64_t>(indexed.sizes().rbegin(), indexed.sizes().rend());
  const auto indexed_strides = std::vector<int64_t>(
      indexed.strides().rbegin(), indexed.strides().rend());
  const auto* indexed_strides_data = indexed_strides.data();
  const auto offset_indexed = OffsetCalculator<1, uindex_t>(
      indexed.dim(), indexed_sizes.data(), &indexed_strides_data);

  const auto loop = [=] C10_DEVICE(int i) {
    const auto offsets = offset_calc.get(i);

    auto& iterated = *reinterpret_cast<scalar_t*>(iterated_ptr + offsets[0]);
    const auto idx = *reinterpret_cast<int64_t*>(idx_ptr + offsets[1]);
    CUDA_KERNEL_ASSERT(
        idx < numel && idx >= -numel &&
        "MusaTakePutKernel() index out of bounds");
    index_t offset = static_cast<index_t>(idx);
    if (offset < 0) {
      offset += numel;
    }
    if (!is_contiguous) {
      offset = offset_indexed.get(offset)[0];
    }

    f(iterated, offset);
  };
  LaunchKernel<LAUNCH_SIZE_ND, LAUNCH_BOUND2>(iter.numel(), loop);
}

void PutKernel(
    TensorIterator& iter,
    const TensorBase& output,
    const bool accumulate) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "put_musa",
      [&] {
        // Cannot use `OpaqueType`, as we need the actual type for
        // `fastSpecializedgpuAtomicAdd`
        AT_DISPATCH_INDEX_TYPES(
            musa::detail::canUse32BitIndexMath(output) ? ScalarType::Int
                                                       : ScalarType::Long,
            "put_musa_index",
            [&] {
              auto* __restrict__ indexed_ptr =
                  output.template data_ptr<scalar_t>();
              if (accumulate) {
                index_t numel = output.numel();
                MusaTakePutKernel<scalar_t, index_t>(
                    iter,
                    output,
                    [numel, indexed_ptr] __device__(
                        scalar_t & iterated, const index_t offset) {
                      fastSpecializedAtomicAdd(
                          indexed_ptr, offset, numel, iterated);
                    });
              } else {
                MusaTakePutKernel<scalar_t, index_t>(
                    iter,
                    output,
                    [indexed_ptr] __device__(
                        scalar_t & iterated, const index_t offset) {
                      indexed_ptr[offset] = iterated;
                    });
              }
            });
      });
}

void TakeKernel(TensorIterator& iter, const TensorBase& input) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "take_musa",
      [&] {
        // Cannot use `OpaqueType`, as Tensor::data_ptr<OpaqueType<N>> is not
        // implemented
        AT_DISPATCH_INDEX_TYPES(
            musa::detail::canUse32BitIndexMath(input) ? ScalarType::Int
                                                      : ScalarType::Long,
            "take_musa_index",
            [&] {
              const auto* __restrict__ indexed_ptr =
                  input.template data_ptr<scalar_t>();
              MusaTakePutKernel<scalar_t, index_t>(
                  iter,
                  input,
                  [indexed_ptr] __device__(
                      scalar_t & iterated, const index_t offset) {
                    iterated = indexed_ptr[offset];
                  });
            });
      });
}

template <typename scalar_t>
void FlipKernelImpl(TensorIterator& iter) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      FlipKernelImpl<scalar_t>(sub_iter);
    }
    return;
  }

  char* const __restrict__ out_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  const char* const __restrict__ in_ptr =
      reinterpret_cast<const char*>(iter.data_ptr(1));

  const auto offset_calc =
      make_offset_calculator<2, /*signed_strides=*/true>(iter);

  const auto loop = [=] C10_DEVICE(const int i) {
    const auto offsets = offset_calc.get(i);
    // offsets can be negative here, but it's fine
    scalar_t* const __restrict__ out_data =
        reinterpret_cast<scalar_t*>(out_ptr + offsets[0]);
    const scalar_t* const __restrict__ in_data =
        reinterpret_cast<const scalar_t*>(in_ptr + offsets[1]);
    *out_data = *in_data;
  };
  LaunchKernel<LAUNCH_SIZE_ND, LAUNCH_BOUND2>(iter.numel(), loop);
}

// The kernels are templated on an opaque, self-aligned type of the correct
// size to avoid redundant kernels for different types of the same size.
template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

void FlipKernel(TensorIterator& iter, const bool quantized) {
  if (quantized) {
    AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
        iter.dtype(), "flip_quantized_musa", [&] {
          using dtype = OpaqueType<sizeof(scalar_t)>;
          FlipKernelImpl<dtype>(iter);
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "flip_musa",
        [&] {
          using dtype = OpaqueType<sizeof(scalar_t)>;
          FlipKernelImpl<dtype>(iter);
        });
  }
}

template <typename scalar_t>
void index_copy_kernel_impl(
    TensorIterator& iter,
    const int64_t dim,
    const int64_t self_dim_size,
    const int64_t self_dim_stride) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      index_copy_kernel_impl<scalar_t>(
          sub_iter, dim, self_dim_size, self_dim_stride);
    }
    return;
  }

  char* const __restrict__ self_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* const __restrict__ idx_ptr = reinterpret_cast<char*>(iter.data_ptr(1));
  char* const __restrict__ source_ptr =
      reinterpret_cast<char*>(iter.data_ptr(2));

  const auto offset_calc = make_offset_calculator<3>(iter);

  const auto loop = [=] C10_DEVICE(int i) {
    const auto offsets = offset_calc.get(i);

    auto* const __restrict__ self_data =
        reinterpret_cast<scalar_t*>(self_ptr + offsets[0]);
    auto idx = *reinterpret_cast<int64_t*>(idx_ptr + offsets[1]);
    const auto* const __restrict__ source_data =
        reinterpret_cast<scalar_t*>(source_ptr + offsets[2]);
    CUDA_KERNEL_ASSERT(
        idx >= 0 && idx < self_dim_size &&
        "index_copy_(): index out of bounds");

    self_data[idx * self_dim_stride] = *source_data;
  };
  LaunchKernel<LAUNCH_SIZE_ND, LAUNCH_BOUND2>(iter.numel(), loop);
}

static void IndexCopyKernel(
    TensorIterator& iter,
    const int64_t dim,
    const int64_t self_dim_size,
    const int64_t self_dim_stride) {
  // See note [Writing Nondeterministic Operations]
  // Nondeterministic when index contains duplicate entries
  // this kernel will not be called when
  // torch.use_deterministic_algorithms(True)
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      kComplexHalf,
      iter.dtype(),
      "index_copy_musa",
      [&] {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        index_copy_kernel_impl<dtype>(
            iter, dim, self_dim_size, self_dim_stride);
      });
}

REGISTER_MUSA_DISPATCH(index_copy_stub, &IndexCopyKernel);
REGISTER_MUSA_DISPATCH(index_put_stub, &IndexPutKernel);
REGISTER_MUSA_DISPATCH(index_stub, &IndexKernel);
REGISTER_MUSA_DISPATCH(put_stub, &PutKernel);
REGISTER_MUSA_DISPATCH(take_stub, &TakeKernel);
REGISTER_MUSA_DISPATCH(flip_stub, &FlipKernel);

} // namespace native
} // namespace at
