#include <ATen/Config.h>
#include <ATen/core/Generator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
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

} // namespace at::musa
