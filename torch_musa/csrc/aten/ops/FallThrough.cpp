#include <ATen/Config.h>
#include <ATen/core/Generator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_batch_norm_with_update_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/_batch_norm_with_update_native.h>
#include <ATen/ops/glu_backward_jvp_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/glu_backward_jvp_native.h>
#include <ATen/ops/glu_jvp_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/glu_jvp_native.h>
#include <ATen/ops/rrelu_with_noise_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/rrelu_with_noise_native.h>
#endif

namespace at::musa {

std::tuple<Tensor, Tensor> RReluWithNoiseFunctional(
    const Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<at::Generator> generator) {
  return at::compositeexplicitautograd::rrelu_with_noise_functional(
      self, noise, lower, upper, training, std::move(generator));
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
_BatchNormWithUpdateFunctional(
    const Tensor& input,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum,
    double eps) {
  return at::compositeexplicitautograd::_batch_norm_with_update_functional(
      input, weight, bias, running_mean, running_var, momentum, eps);
}

Tensor& GluJvpOut(
    const Tensor& glu,
    const Tensor& x,
    const Tensor& dx,
    int64_t dim,
    Tensor& out) {
  return at::compositeexplicitautograd::glu_jvp_out(out, glu, x, dx, dim);
}

Tensor& GluBwdJvpOut(
    const Tensor& grad_x,
    const Tensor& grad_glu,
    const Tensor& x,
    const Tensor& dgrad_glu,
    const Tensor& dx,
    int64_t dim,
    Tensor& out) {
  return at::compositeexplicitautograd::glu_backward_jvp_out(
      out, grad_x, grad_glu, x, dgrad_glu, dx, dim);
}

} // namespace at::musa
