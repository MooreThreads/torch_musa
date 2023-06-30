#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ops/int_repr_native.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace {

Tensor WrapQuantizedMusaIntRepr(const Tensor& self) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::int_repr_quantized_cuda(self);
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("int_repr", TORCH_FN(WrapQuantizedMusaIntRepr));
}

} // namespace
} // namespace at
