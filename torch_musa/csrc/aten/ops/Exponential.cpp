#include <ATen/ops/exponential_native.h>
#include <torch/library.h>

#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace musa {
Tensor& Exponential_(
    Tensor& self,
    double lambd,
    c10::optional<at::Generator> generator) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::exponential_(self, lambd, generator);
}

ADVANCED_REGISTER(aten, PrivateUse1, "exponential_", Exponential_)

} // namespace musa
} // namespace at
