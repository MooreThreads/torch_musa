#include <ATen/Config.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {
::std::tuple<Tensor, Tensor> NativeDropout(
    const Tensor& input,
    double p,
    c10::optional<bool> train) {
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of NativeDropout must be MUSA, ",
      "but now is ",
      input.device());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of NativeDropout only support Float32, ",
      "but now it is ",
      input.scalar_type());

  c10::musa::MUSAGuard device_guard(input.device());
  if (input.numel() == 0) {
    return std::make_tuple(input, at::empty_like(input, input.options()));
  }
  // short-cut for train == false
  if (train.has_value() && !train.value()) {
    return std::make_tuple(
        input.clone(),
        at::ones_like(
            input,
            input.options().dtype(c10::CppTypeToScalarType<bool>::value)));
  }
  // short-cut
  if (p == 1) {
    // native_dropout is in derivatives.yaml, so we don't need to add data
    // dependency from output to input for autograd
    auto ret = at::zeros_like(input);
    auto mask = at::zeros_like(
        input, input.options().dtype(c10::CppTypeToScalarType<bool>::value));
    return std::tuple<Tensor, Tensor>(ret, mask);
  }

  Tensor mask = at::empty_like(
      input, input.options().dtype(c10::CppTypeToScalarType<bool>::value));
  Tensor output = at::empty_like(input);
  muHandle& h = GetMudnnHandle();
  auto musa_input = CreateMUTensor(input);
  auto musa_output = CreateMUTensor(output);
  auto musa_mask = CreateMUTensor(mask);

  ::musa::dnn::Dropout dropout;
  CHECK_MUDNN_STATUS(dropout.SetP(p), "SetP");
  CHECK_MUDNN_STATUS(
      dropout.RunDropout(h, musa_output, musa_input, musa_mask), "RunDropout");
  return std::make_tuple(output, mask);
}

Tensor NativeDropoutBackward(
    const Tensor& grad_output,
    const Tensor& mask,
    double scale) {
  TORCH_CHECK(
      grad_output.device().type() == kMUSA,
      "Device of input tensor of NativeDropoutBackward must be MUSA,",
      " but now is ",
      grad_output.device());
  TORCH_CHECK(
      grad_output.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of NativeDropoutBackward only support",
      " Float32, but now it is ",
      grad_output.scalar_type());
  TORCH_CHECK(
      mask.device().type() == kMUSA,
      "Device of mask tensor of NativeDropoutBackward must be MUSA,",
      " but now is ",
      mask.device());
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Bool,
      "Dtype of mask tensor of NativeDropoutBackward only support",
      " Bool, but now it is ",
      mask.scalar_type());
  c10::musa::MUSAGuard device_guard(grad_output.device());
  Tensor output = at::empty_like(grad_output);
  muHandle& h = GetMudnnHandle();
  auto musa_grad_output = CreateMUTensor(grad_output);
  auto musa_mask = CreateMUTensor(mask);
  auto musa_output = CreateMUTensor(output);

  ::musa::dnn::Dropout dropout;
  dropout.SetScale(scale);
  CHECK_MUDNN_STATUS(
      dropout.RunDropoutBwd(h, musa_output, musa_grad_output, musa_mask),
      "RunDropoutBwd");
  return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("native_dropout", &NativeDropout);
  m.impl("native_dropout_backward", &NativeDropoutBackward);
}

} // namespace musa
} // namespace at
