#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

at::Tensor& RandomFrom(
    at::Tensor& self,
    int64_t from,
    c10::optional<int64_t> to,
    c10::optional<at::Generator> generator) {
  const c10::musa::MUSAGuard device_guard(self.device());
  return at::native::random_(self, from, to, generator);
}

at::Tensor& RandomTo(
    at::Tensor& self,
    int64_t to,
    c10::optional<at::Generator> generator) {
  const c10::musa::MUSAGuard device_guard(self.device());
  return at::native::random_(self, to, generator);
}

at::Tensor& Random(at::Tensor& self, c10::optional<at::Generator> generator) {
  const c10::musa::MUSAGuard device_guard(self.device());
  return at::native::random_(self, generator);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("random_.from", &RandomFrom);
  m.impl("random_.to", &RandomTo);
  m.impl("random_", &Random);
}

} // namespace musa
} // namespace at
