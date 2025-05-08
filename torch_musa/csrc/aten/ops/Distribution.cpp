#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

Tensor& BernoulliFloat(
    Tensor& self,
    double p,
    c10::optional<at::Generator> generator) {
  c10::musa::MUSAGuard device_guard(self.device());
  if (at::musa::getMUSAArch() >= 210) {
    return at::native::bernoulli_(self, p, generator);
  }
  auto cpu_tensor =
      at::empty(self.sizes(), self.options().device(DeviceType::CPU));
  auto cpu_result = at::native::bernoulli_(cpu_tensor, p, generator);
  self.copy_(cpu_result);
  return self;
}
} // namespace musa
} // namespace at
