#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

Tensor& BernoulliFloat(
    Tensor& self,
    double p,
    c10::optional<at::Generator> generator) {
  auto cpu_tensor =
      ::at::empty(self.sizes(), self.options().device(DeviceType::CPU));
  auto cpu_result = at::native::bernoulli_(cpu_tensor, p, generator);
  self.copy_(cpu_result);
  return self;
}

Tensor& Normal(
    Tensor& self,
    double mean,
    double std,
    c10::optional<Generator> gen) {
  Device self_device = self.device();
  self = self.to("cpu"); // assign to reference ?
  self = at::native::normal_(self, mean, std, gen);
  self = self.to(self_device);
  return self;
}

Tensor& Uniform(
    Tensor& self,
    double from,
    double to,
    c10::optional<Generator> gen) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::uniform_(self, from, to, gen);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("bernoulli_.float", &BernoulliFloat);
  m.impl("normal_", &Normal);
  m.impl("uniform_", &Uniform);
}

} // namespace musa
} // namespace at
