#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/pad.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#endif

#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/ops/Reduce.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <algorithm>
#include <iterator>
#include <numeric>

namespace at {
namespace native { // this namespace is used to impl logsumexp stub only
namespace {

constexpr int THREAD_PER_BLOCK = 512;
constexpr int WARP_SIZE = 32;
constexpr int WARP_PER_BLOCK = THREAD_PER_BLOCK / WARP_SIZE;

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
__inline__ __device__ T Inf();

template <>
__inline__ __device__ float Inf() {
  return MUSART_INF_F;
}

template <>
__inline__ __device__ double Inf() {
  return MUSART_INF;
}

template <>
__inline__ __device__ float16_t Inf() {
  return MUSART_INF_F;
}

template <>
__inline__ __device__ bfloat16_t Inf() {
  return MUSART_INF_F;
}

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) {
    return max(a, b);
  }
};

template <template <typename> class ReduceOp, typename T, int64_t warp_size>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int mask = warp_size / 2; mask > 0; mask >>= 1) {
    val = ReduceOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
};

template <typename T, int num_warp>
__global__ void LogSumExpKernel(
    T* result,
    const T* self,
    const int64_t reduce_size,
    const int64_t elements,
    const int64_t act_warps) {
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t idy = blockIdx.y;
  int64_t index = idx + idy * blockDim.x * gridDim.x;
  int64_t tid = threadIdx.x;
  __shared__ T smem[THREAD_PER_BLOCK];
  __shared__ T warp_max[num_warp];

  // load data from global mem to shared mem
  if (tid < blockDim.x && idx < elements) {
    smem[tid] = self[idx];
  } else {
    smem[tid] = 0.f;
  }
  __syncthreads();

  // find max value of a warp
  // NOTICE: musa warp should be 128, since we utilize __shfl_xor_sync
  // function, which is aligned with 32 threads as a warp.
  int warp_id = tid / WARP_SIZE;
  warp_max[warp_id] = WarpAllReduce<MaxOp, T, WARP_SIZE>(smem[tid]);
  __syncthreads();

  // find the max value of a block
  T max_val = -Inf<T>();
#pragma unroll
  for (int i = 0; i < act_warps; ++i) {
    max_val = max(warp_max[i], max_val);
  }

  // act exponential and avoid numerical overflow
  smem[tid] = __expf(smem[tid] - max_val);
  __syncthreads();

  // reduction in shared mem
#pragma unroll
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && tid + s < reduce_size) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }

  // write result to global memory
  if (tid == 0) {
    result[blockIdx.x * gridDim.y + blockIdx.y] =
        __logf(smem[tid]) + (float)max_val;
  }
}

template <typename T>
void LogSumExpImpl(Tensor& result, const Tensor& self, const int dim) {
  auto stream = c10::musa::getCurrentMUSAStream();
  IntArrayRef shape = self.sizes();
  Tensor self_tensor = self;
  // for non-last-dim reduce, we permute the reduction dim to the last
  if (dim != shape.size() - 1) {
    auto range = c10::irange(self_tensor.dim());
    std::vector<int64_t> range_vec(range.begin(), range.end());
    range_vec.erase(range_vec.begin() + dim);
    range_vec.push_back(dim);
    self_tensor = musa::FormatContiguous(
        self_tensor.permute(range_vec), at::MemoryFormat::Contiguous);
  }

  int64_t reduce_size = std::min<uint32_t>(shape[dim], THREAD_PER_BLOCK);
  uint32_t block_x =
      std::max<uint32_t>(NextPowerOf2((uint32_t)reduce_size), WARP_SIZE);

  // for tensor reduce shape < THREAD_PER_BLOCK, we pad it to make the reduce
  // shape match the block size
  if (block_x > shape[dim]) {
    uint32_t pad_num = block_x - shape[dim];
    self_tensor = at::pad(self_tensor, {0, pad_num}, "constant", 0);
  }

  int64_t numel = self_tensor.numel();
  int64_t warps = block_x / WARP_SIZE;
  uint32_t grids = (numel + block_x - 1) / block_x;
  uint32_t grid_y = (block_x + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  uint32_t grid_x = grids / grid_y;

  dim3 grid_size(grid_x, grid_y, 1);
  dim3 block_size(block_x, 1, 1);

  // do logsumexp directly
  LogSumExpKernel<T, WARP_PER_BLOCK><<<grid_size, block_size, 0, stream>>>(
      static_cast<T*>(result.data_ptr()),
      static_cast<T*>(self_tensor.data_ptr()),
      reduce_size,
      numel,
      warps);
}

void LogSumExpMusa(Tensor& result, const Tensor& self, int64_t dim) {
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Half ||
          self.scalar_type() == at::ScalarType::BFloat16 ||
          self.scalar_type() == at::ScalarType::Double,
      "musa logsumexp currently only supports float16, bfloat16, float32 and doulbe dtype");
  TORCH_CHECK(
      self.size(dim) <= THREAD_PER_BLOCK,
      "musa logsumexp currently doesn't support reduce size greater than ",
      THREAD_PER_BLOCK);
  auto data_type = self.scalar_type();
  switch (data_type) {
    case at::ScalarType::Float:
      LogSumExpImpl<float>(result, self, dim);
      break;
    case at::ScalarType::Double:
      LogSumExpImpl<double>(result, self, dim);
      break;
    case at::ScalarType::Half:
      LogSumExpImpl<float16_t>(result, self, dim);
      break;
#if TORCH_MUSA_ARCH > 210
    case at::ScalarType::BFloat16:
      LogSumExpImpl<bfloat16_t>(result, self, dim);
      break;
#endif
    default:
      break;
  }
}

} // anonymous namespace

REGISTER_MUSA_DISPATCH(logsumexp_stub, &LogSumExpMusa);

} // namespace native
} // namespace at
