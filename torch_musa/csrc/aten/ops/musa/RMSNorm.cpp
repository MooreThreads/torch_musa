#include <ATen/ATen.h>

#include "torch_musa/csrc/aten/ops/musa/RMSNorm.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/rms_norm_forward.h>
#endif

namespace at {
namespace musa {

::std::tuple<at::Tensor, at::Tensor> RMSNormForward(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    double eps) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const auto weight = *weight_maybe_owned;
  const int normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      input.device().type() == at::DeviceType::PrivateUse1,
      "Device of input tensor of RMSNorm must be MUSA, but now is ",
      input.device());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float ||
          input.scalar_type() == at::ScalarType::Half ||
          input.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of input tensor of RMSNorm only support Float32, Half and BFloat16, but now it is ",
      input.scalar_type());
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(weight.defined(), "This kernel neeeds weight must has value!");
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() ||
          (weight.scalar_type() == at::ScalarType::Float ||
           weight.scalar_type() == at::ScalarType::Half ||
           weight.scalar_type() == at::ScalarType::BFloat16),
      "Dtype of weight tensor of RMSNorm only support Float32, Half and BFloat16",
      "but now it is ",
      weight.scalar_type());
  // Device guard
  c10::musa::MUSAGuard device_guard(input.device());
  // Generate ouput && square
  at::Tensor contiguous_input = input.contiguous();
  at::Tensor gamma;
  if (weight.defined()) {
    gamma = weight.contiguous();
  }

  auto output = at::empty_like(contiguous_input);
  int inner = 1, outter = 1;
  int dims_diff = contiguous_input.dim() - normalized_ndim;
  for (int i = 0; i < normalized_ndim; i++) {
    TORCH_INTERNAL_ASSERT(
        input.size(i + dims_diff) == normalized_shape[i],
        "Mismatched normalized_shape with input tensor's shape");
    inner *= normalized_shape[i];
  }
  for (int i = 0; i < dims_diff; i++) {
    outter *= contiguous_input.size(i);
  }

  // use fp32 as mean calculation buffer for float dtype
  auto invvar_dtype = (input.scalar_type() == at::ScalarType::Half ||
                       input.scalar_type() == at::ScalarType::BFloat16)
      ? at::ScalarType::Float
      : input.scalar_type();
  auto invvar =
      at::empty({outter}, contiguous_input.options().dtype(invvar_dtype));
  // kernel computation
  musa_rms_norm(
      contiguous_input,
      invvar,
      output,
      gamma,
      inner, // -> n2
      outter, // -> n1
      normalized_shape,
      eps);
  return {output, invvar};
}

::std::tuple<at::Tensor, at::Tensor> RMSNormBackward(
    const at::Tensor& grad_out,
    const at::Tensor& invvar,
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    double eps) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const auto weight = *weight_maybe_owned;
  c10::musa::MUSAGuard device_guard(input.device());
  at::Tensor contiguous_input = input.contiguous();
  auto contiguous_grad_output = grad_out.contiguous();
  auto grad_input = at::empty_like(contiguous_input);
  at::Tensor grad_gamma, gamma;
  if (weight.defined()) {
    gamma = weight.contiguous();
    grad_gamma = at::empty_like(gamma);
  }
  int inner = 1, outter = 1;
  const int normalized_ndim = normalized_shape.size();
  int dims_diff = contiguous_input.dim() - normalized_ndim;
  for (int i = 0; i < normalized_ndim; i++) {
    TORCH_INTERNAL_ASSERT(
        input.size(i + dims_diff) == normalized_shape[i],
        "Mismatched normalized_shape with input tensor's shape");
    inner *= normalized_shape[i];
  }
  for (int i = 0; i < dims_diff; i++) {
    outter *= contiguous_input.size(i);
  }
  musa_rms_norm_backward(
      contiguous_grad_output,
      invvar,
      contiguous_input,
      gamma,
      grad_input,
      grad_gamma,
      normalized_shape,
      outter, // N in [N ,D]
      inner, // D in [N,D]
      eps);
  return {grad_input, grad_gamma};
}

} // namespace musa
} // namespace at
