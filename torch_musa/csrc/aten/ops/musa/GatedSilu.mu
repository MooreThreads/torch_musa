
#include <ATen/core/Scalar.h>
#include <ATen/ops/empty.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace musa {

#define MAX_THREADS 1024

template <typename T>
__device__ __forceinline__ T silu(const T& x) {
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename T>
__global__ void gated_silu_kernel(
    T* out_ptr,
    const T* input_ptr,
    const int hidden_size,
    const int element_num) {
  const int cur_element_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (cur_element_idx >= element_num) {
    return;
  }
  const int input_x_idx = cur_element_idx / hidden_size * 2 * hidden_size +
      cur_element_idx % hidden_size;
  const int input_y_idx = input_x_idx + hidden_size;

  const T x = __ldg(&input_ptr[input_x_idx]);
  const T y = __ldg(&input_ptr[input_y_idx]);
  out_ptr[cur_element_idx] = silu(x) * y;
}

void gated_silu_output(const at::Tensor& input, at::Tensor& out) {
  TORCH_CHECK(
      (input.scalar_type() == at::ScalarType::Float ||
       input.scalar_type() == at::ScalarType::Half),
      "Dtype of input tensor of gated silu only support fp32 / fp16, but now it is ",
      input.scalar_type());
  TORCH_CHECK(
      input.dim() > 1,
      "Dim of input tensor of gated silu should greater than 1",
      input.scalar_type());
  TORCH_CHECK(
      input.is_contiguous(),
      "gated silu only support contiguous tensor",
      input.scalar_type());

  int hidden_size = out.size(out.dim() - 1);
  int element_num = out.numel();

  int cur_device = -1;
  TORCH_MUSA_CHECK(musaGetDevice(&cur_device));

  size_t block_num = (element_num - 1) / MAX_THREADS + 1;
  dim3 grid(block_num);
  dim3 block(MAX_THREADS);

  const musaStream_t stream = c10::musa::getCurrentMUSAStream(cur_device);
  AT_DISPATCH_ALL_MTGPU_TYPES_AND_HALF(
      input.scalar_type(), "gated_silu_kernel", [&] {
        gated_silu_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            hidden_size,
            element_num);
      });
}

at::Tensor gated_silu(const at::Tensor& input) {
  c10::musa::MUSAGuard device_guard(input.device());
  std::vector<int64_t> size;
  for (int i = 0; i < input.dim(); ++i) {
    int64_t cur_size = input.size(i);
    if (i == input.dim() - 1) {
      cur_size = cur_size / 2;
    }
    size.push_back(cur_size);
  }
  auto output = at::empty(
      size, input.options().memory_format(at::MemoryFormat::Contiguous));
  gated_silu_output(input, output);
  return output;
}

} // namespace musa
} // namespace at
