#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/randint.h>
#include <ATen/ops/randint_native.h>
#include <ATen/ops/random_native.h>
#endif

#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/musa_lazy_init.h"

namespace at {
namespace musa {

at::Tensor& RandomFrom(
    at::Tensor& self,
    int64_t from,
    c10::optional<int64_t> to,
    c10::optional<at::Generator> generator) {
  torch::utils::musa_lazy_init();
  const c10::musa::MUSAGuard device_guard(self.device());
  return at::native::random_(self, from, to, generator);
}

at::Tensor& RandomTo(
    at::Tensor& self,
    int64_t to,
    c10::optional<at::Generator> generator) {
  torch::utils::musa_lazy_init();
  const c10::musa::MUSAGuard device_guard(self.device());
  return at::native::random_(self, to, generator);
}

at::Tensor& Random(at::Tensor& self, c10::optional<at::Generator> generator) {
  torch::utils::musa_lazy_init();
  const c10::musa::MUSAGuard device_guard(self.device());
  return at::native::random_(self, generator);
}

} // namespace musa
} // namespace at
