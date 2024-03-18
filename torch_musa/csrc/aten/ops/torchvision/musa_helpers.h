#pragma once
#include "torch_musa/csrc/aten/ops/TensorFactory.h"

namespace vision {
namespace ops {

#define MUSA_1D_KERNEL_LOOP_T(i, n, index_t)                         \
  for (index_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))

#define MUSA_1D_KERNEL_LOOP(i, n) MUSA_1D_KERNEL_LOOP_T(i, n, int)

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

} // namespace ops
} // namespace vision
