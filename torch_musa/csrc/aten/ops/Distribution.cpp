#pragma GCC diagnostic push

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#pragma GCC diagnostic pop

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace native {
namespace musa {

Tensor& BernoulliFloat(
    Tensor& self,
    double p,
    c10::optional<at::Generator> generator) {
  auto cpu_tensor =
      ::at::empty(self.sizes(), self.options().device(DeviceType::CPU));
  auto cpu_result = bernoulli_(cpu_tensor, p, generator);
  self.copy_(cpu_result);
  return self;
}

Tensor& Normal(
    Tensor& self,
    double mean,
    double std,
    c10::optional<Generator> gen) {
  self = self.to("cpu");
  self = normal_(self, mean, std, gen);
  self = self.to("musa");
  return self;
}

Tensor& Uniform(
    Tensor& self,
    double from,
    double to,
    c10::optional<Generator> gen) {
  auto cpu_tensor =
      ::at::empty(self.sizes(), self.options().device(DeviceType::CPU));
  auto cpu_result = uniform_(cpu_tensor, from, to, gen);
  self.copy_(cpu_result);
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("bernoulli_.float", &BernoulliFloat);
  m.impl("normal_", &Normal);
  m.impl("uniform_", &Uniform);
}

} // namespace musa
} // namespace native
} // namespace at
