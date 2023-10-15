#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#endif

#include "torch_musa/csrc/aten/ops/Reduce.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <algorithm>
#include <iterator>
#include <numeric>

namespace at {
namespace native { // this namespace is used to impl logsumexp stub only
namespace {

constexpr int THREAD_PER_BLOCK = 512;

unsigned int NextPowerOf2(const unsigned int target) {
  int temp = target - 1;
  temp |= temp >> 1;
  temp |= temp >> 2;
  temp |= temp >> 4;
  temp |= temp >> 8;
  temp |= temp >> 16;
  return (temp < 0) ? 1 : temp + 1;
}

template <typename T>
__device__ __forceinline__ void warpReduce(T* data, const int tid) {
  data[tid] += data[tid + 128];
  data[tid] += data[tid + 64];
  data[tid] += data[tid + 32];
  data[tid] += data[tid + 16];
  data[tid] += data[tid + 8];
  data[tid] += data[tid + 4];
  data[tid] += data[tid + 2];
  data[tid] += data[tid + 1];
}

template <typename T>
__global__ void LogSumExpKernel(
    T* result,
    const T* self,
    const int64_t reduce_size) {
  int64_t idx;
  __shared__ T smem[THREAD_PER_BLOCK];

  // load data from global mem to shared mem
  int64_t tid = threadIdx.x;
  if (tid < reduce_size) {
    idx = tid + blockIdx.x * reduce_size;
    smem[tid] = __expf(self[idx]);
  } else {
    smem[tid] = 0.f;
  }
  __syncthreads();

  // reduction in shared mem, e.g musa warp has 128 threads
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }

  // if (tid < 128) {
  //   warpReduce<T>(smem, tid);
  // }

  if (tid == 0) {
    result[blockIdx.x] = __logf(smem[tid]);
  }
}

template <typename T>
void LogSumExpImpl(Tensor& result, const Tensor& self, const int dim) {
  auto stream = c10::musa::getCurrentMUSAStream();
  int64_t numel = self.numel();
  IntArrayRef shape = self.sizes();

  int64_t reduce_size = std::min<int64_t>(shape[dim], THREAD_PER_BLOCK);
  uint32_t block_x = NextPowerOf2((unsigned int)reduce_size);
  uint32_t grid_x = (numel + reduce_size - 1) / reduce_size;

  dim3 grid_size(grid_x, 1, 1);
  dim3 block_size(block_x, 1, 1);

  Tensor self_tensor = self;

  // for non-last-dim reduce, we permute the reduce dim to the last dim
  if (dim != shape.size() - 1) {
    self_tensor = self_tensor.transpose(dim, -1).contiguous();
  }

  LogSumExpKernel<T>
      <<<grid_size, block_size, THREAD_PER_BLOCK * sizeof(T), stream>>>(
          static_cast<T*>(result.data_ptr()),
          static_cast<T*>(self_tensor.data_ptr()),
          reduce_size);

  musaDeviceSynchronize();
}

void LogSumExpMusa(Tensor& result, const Tensor& self, IntArrayRef dims) {
  std::vector<int> dim_int(dims.begin(), dims.end());
  TORCH_CHECK(
      dim_int.size() == 1 && dim_int[0] >= 0,
      "musa logsumexp doesn't support multi-dim yet ",
      "and dim should be positive, which is now: ",
      dim_int[0]);
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Half,
      "musa logsumexp currently only supports float32/half dtype");
  auto data_type = self.scalar_type();
  switch (data_type) {
    case at::ScalarType::Float:
      LogSumExpImpl<float>(result, self, dim_int[0]);
      break;
    case at::ScalarType::Half:
      LogSumExpImpl<__half>(result, self, dim_int[0]);
    default:
      break;
  }
}

} // anonymous namespace

REGISTER_MUSA_DISPATCH(logsumexp_stub, &LogSumExpMusa);

} // namespace native
} // namespace at
