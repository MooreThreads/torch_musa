#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <mudnn.h>
#include <torch/library.h>
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

using mtHandle = ::musa::dnn::Handle;

static inline void CheckInputTensor(
    const Tensor& input1,
    const Tensor& input2) {
  TORCH_CHECK(input1.device().type() == kMUSA, "device type must be musa");
  TORCH_CHECK(
      input1.device().type() == input2.device().type(),
      "device type of input tensor must be same");
  TORCH_CHECK(
      input1.scalar_type() == at::ScalarType::Float ||
          input1.scalar_type() == at::ScalarType::Half ||
          input1.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of input tensor only support Float32 and Float16/Bfloat16, ",
      "but now is ",
      input1.scalar_type());
}

Tensor& RopeOut(
    const Tensor& input,
    const Tensor& freqs_cis,
    bool rotary_interleaved,
    bool batch_first,
    Tensor& output) {
#if defined(MUDNN_VERSION) && MUDNN_VERSION >= 2800

  c10::musa::MUSAGuard device_guard(input.device());
  if C10_UNLIKELY (input.numel() == 0) {
    TORCH_CHECK(
        output.numel() == 0, "Rope with empty input needs empty output");
    return output;
  }

  if (!output.sizes().equals(input.sizes())) {
    TORCH_CHECK(
        output.numel() == input.numel(),
        "rope result must be same elements with input");
    output.resize_(input.sizes());
  }

  CheckInputTensor(input, freqs_cis);
  Tensor input_contiguous = input;
  if (!IsLastDimContiguous(input)) {
    input_contiguous = input.contiguous();
  }

  Tensor freqs_cis_contiguous = freqs_cis.contiguous();
  Tensor output_contiguous = output.contiguous();

  auto mt_input = CreateMUTensor(input_contiguous);
  auto mt_output = CreateMUTensor(output_contiguous);
  auto mt_freqs_cis = CreateMUTensor(freqs_cis_contiguous);

  mtHandle& handler = GetMudnnHandle();
  ::musa::dnn::Rope rope;

  CHECK_MUDNN_STATUS(
      rope.SetRotaryInterleaved(rotary_interleaved), "SetRotaryInterleaved");
  CHECK_MUDNN_STATUS(rope.SetBatchFirst(batch_first), "SetBatchFirst");
  CHECK_MUDNN_STATUS(
      rope.Run(handler, mt_output, mt_input, mt_freqs_cis), "Run");

  if (!output.is_same(output_contiguous)) {
    output.copy_(output_contiguous);
  }
#else
  TORCH_CHECK(false, "RoPE only support for MUDNN_VERSION >= 2800");
#endif
  return output;
}

Tensor Rope(
    const Tensor& input,
    const Tensor& freqs_cis,
    bool rotary_interleaved,
    bool batch_first) {
  Tensor output = at::empty_like(input, at::MemoryFormat::Contiguous);
  return RopeOut(input, freqs_cis, rotary_interleaved, batch_first, output);
}

Tensor& RopeBackwardOut(
    const Tensor& grad_output,
    const Tensor& freqs_cis,
    bool rotary_interleaved,
    bool batch_first,
    Tensor& grad_input) {
#if defined(MUDNN_VERSION) && MUDNN_VERSION >= 2800
  c10::musa::MUSAGuard device_guard(grad_output.device());

  TORCH_CHECK(
      grad_input.sizes() == grad_output.sizes(),
      "Rope grad input & output sizes don't match");
  if C10_UNLIKELY (grad_output.numel() == 0) {
    return grad_input;
  }

  CheckInputTensor(grad_output, grad_input);
  Tensor grad_output_contiguous = grad_output;

  if (!IsLastDimContiguous(grad_output)) {
    grad_output_contiguous = grad_output.contiguous();
  }

  Tensor freqs_cis_contiguous = freqs_cis.contiguous();
  Tensor grad_input_contiguous = grad_input.contiguous();
  auto mt_grad_output = CreateMUTensor(grad_output_contiguous);
  auto mt_freqs_cis = CreateMUTensor(freqs_cis_contiguous);
  auto mt_grad_input = CreateMUTensor(grad_input_contiguous);

  mtHandle& handler = GetMudnnHandle();
  ::musa::dnn::Rope rope;

  CHECK_MUDNN_STATUS(
      rope.SetRotaryInterleaved(rotary_interleaved), "SetRotaryInterleaved");
  CHECK_MUDNN_STATUS(rope.SetBatchFirst(batch_first), "SetBatchFirst");
  CHECK_MUDNN_STATUS(
      rope.RunBwd(handler, mt_grad_input, mt_grad_output, mt_freqs_cis),
      "RunBwd");
  if (!grad_input.is_same(grad_input_contiguous)) {
    grad_input.copy_(grad_input_contiguous);
  }
#else
  TORCH_CHECK(false, "RopeBackward only support for MUDNN_VERSION >= 2800");
#endif
  return grad_input;
}

Tensor RopeBackward(
    const Tensor& grad_output,
    const Tensor& freqs_cis,
    bool rotary_interleaved,
    bool batch_first) {
  Tensor grad_input = at::empty_like(grad_output, at::MemoryFormat::Contiguous);
  return RopeBackwardOut(
      grad_output, freqs_cis, rotary_interleaved, batch_first, grad_input);
}

} // namespace musa
} // namespace at
