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
  c10::musa::MUSAGuard device_guard(self.device());
#if TORCH_MUSA_ARCH >= 210
  return at::native::bernoulli_(self, p, generator);
#else
  auto cpu_tensor =
      at::empty(self.sizes(), self.options().device(DeviceType::CPU));
  auto cpu_result = at::native::bernoulli_(cpu_tensor, p, generator);
  self.copy_(cpu_result);
  return self;
#endif
}

Tensor& BernoulliTensor(
    Tensor& self,
    const Tensor& p,
    c10::optional<at::Generator> generator) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::bernoulli_(self, p, generator);
}

Tensor& BernoulliOut(
    const Tensor& self,
    c10::optional<at::Generator> generator,
    Tensor& out) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::bernoulli_out(self, generator, out);
}

Tensor& Normal(
    Tensor& self,
    double mean,
    double std,
    c10::optional<Generator> gen) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::normal_(self, mean, std, gen);
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
  m.impl("bernoulli_.Tensor", &BernoulliTensor);
  m.impl("bernoulli.out", &BernoulliOut);
  m.impl("normal_", &Normal);
  m.impl("uniform_", &Uniform);
}

} // namespace musa
} // namespace at
