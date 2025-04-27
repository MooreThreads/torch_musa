#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <ATen/ops/empty.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

static inline void CheckInputTensor(const Tensor& input) {
  TORCH_CHECK(
      input.device().type() == kMUSA, "device type of input must be musa");
  TORCH_CHECK(
      input.scalar_type() == ScalarType::Float ||
          input.scalar_type() == ScalarType::Half ||
          input.scalar_type() == ScalarType::BFloat16,
      "SwiGLu only supports for float32/float16/bfloat16");
}

static inline void CheckBwdInputTensor(
    const Tensor& grad_out,
    const Tensor& input) {
  CheckInputTensor(input);
  TORCH_CHECK(
      grad_out.device().type() == kMUSA,
      "device type of grad_out must be musa");
}

static inline std::vector<int64_t> get_output_sizes(const Tensor& input) {
  std::vector<int64_t> result;

  int dim_len = input.dim();
  for (int i = 0; i < dim_len; ++i) {
    int64_t cur_size = input.size(i);
    result.emplace_back(i == dim_len - 1 ? cur_size / 2 : cur_size);
  }
  return result;
}

Tensor& SwishGluOut(const Tensor& self, Tensor& output) {
#if defined(MUDNN_VERSION) && MUDNN_VERSION >= 2800

  if C10_UNLIKELY (self.numel() == 0) {
    TORCH_CHECK(
        output.numel() == 0,
        "swishglu with empty input, output must be a empty tensor");
    return output;
  }
  c10::musa::MUSAGuard device_guard(self.device());
  CheckInputTensor(self);
  auto output_sizes = get_output_sizes(self);
  if (!output.sizes().equals(output_sizes)) {
    output.resize_(output_sizes);
  }

  Tensor input_contiguous = self.contiguous();
  Tensor output_contiguous = output.contiguous();

  auto mt_input = CreateMUTensor(input_contiguous);
  auto mt_output = CreateMUTensor(output_contiguous);

  muHandle& h = GetMudnnHandle();

  ::musa::dnn::SwiGlu op;
  CHECK_MUDNN_STATUS(op.Run(h, mt_output, mt_input), "Run");

  if (!output.is_same(output_contiguous)) {
    output.copy_(output_contiguous);
  }
#else
  TORCH_CHECK(false, "SwiGLU is not supported yet");
#endif
  return output;
}

Tensor SwishGlu(const Tensor& self) {
  auto output_sizes = get_output_sizes(self);
  Tensor output = at::empty(
      output_sizes, self.options().memory_format(at::MemoryFormat::Contiguous));

  return SwishGluOut(self, output);
}

Tensor& SwishGluBackwardOut(
    const Tensor& grad_output,
    const Tensor& input,
    Tensor& grad_input) {
#if defined(MUDNN_VERSION) && MUDNN_VERSION >= 2800

  if C10_UNLIKELY (grad_output.numel() == 0) {
    TORCH_CHECK(
        grad_input.numel() == 0, "SwishGlu with empty grad needs empty input");
    return grad_input;
  }
  c10::musa::MUSAGuard device_guard(grad_output.device());
  CheckBwdInputTensor(input, grad_input);

  Tensor grad_output_contiguous = grad_output.contiguous();
  Tensor input_contiguous = input.contiguous();
  Tensor grad_input_contiguous = grad_input.contiguous();

  auto mt_grad_output = CreateMUTensor(grad_output_contiguous);
  auto mt_input = CreateMUTensor(input_contiguous);
  auto mt_grad_input = CreateMUTensor(grad_input_contiguous);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::SwiGlu op;
  CHECK_MUDNN_STATUS(
      op.RunBwd(h, mt_grad_input, mt_input, mt_grad_output), "RunBwd");

  if (!grad_input.is_same(grad_input_contiguous)) {
    grad_input.copy_(grad_input_contiguous);
  }
#else
  TORCH_CHECK(false, "SwishGLUBackward is not supported yet");
#endif
  return grad_input;
}

Tensor SwishGluBackward(const Tensor& grad_output, const Tensor& input) {
  auto grad_input = at::empty_like(input, MemoryFormat::Contiguous);
  return SwishGluBackwardOut(grad_output, input, grad_input);
}

} // namespace musa
} // namespace at
