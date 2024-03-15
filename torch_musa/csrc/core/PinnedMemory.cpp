#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#else
#include <ATen/ops/_pin_memory_ops.h>
#include <ATen/ops/is_pinned_ops.h>
#endif

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace musa {

bool IsPinnedCPU(const at::Tensor& self, c10::optional<at::Device> device) {
  // Only CPU tensors can be pinned
  if (!self.is_cpu()) {
    return false;
  }
  DispatchKeySet _dk = c10::DispatchKeySet(c10::computeDispatchKey(
      c10::nullopt, self.layout(), device.value_or(kMUSA)));
  return at::_ops::is_pinned::redispatch(_dk, self, device);
}

at::Tensor PinMemoryCPU(
    const at::Tensor& self,
    c10::optional<at::Device> device) {
  TORCH_CHECK(
      self.device().is_cpu(),
      "cannot pin '",
      self.toString(),
      "' only dense CPU tensors can be pinned");
  DispatchKeySet _dk = c10::DispatchKeySet(c10::computeDispatchKey(
      c10::nullopt, self.layout(), device.value_or(kMUSA)));
  if (self.is_nested()) {
    constexpr auto nested_key_set = c10::DispatchKeySet(
        {c10::DispatchKey::NestedTensor,
         c10::DispatchKey::AutogradNestedTensor});
    _dk = _dk.add(self.key_set() & nested_key_set);
  }
  return at::_ops::_pin_memory::redispatch(_dk, self, device);
}

OVERRIDE_SELECTIVE_OPERATOR_REGISTER_WITHOUT_WARNING(
    "aten::is_pinned",
    IsPinnedCPU)
OVERRIDE_SELECTIVE_OPERATOR_REGISTER_WITHOUT_WARNING(
    "aten::_pin_memory",
    PinMemoryCPU)

} // namespace musa
} // namespace at
