#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

Tensor& Fill(Tensor& self, const Scalar& value) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::fill_(self, value);
}

Tensor& Zero_(Tensor& self) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::zero_(self);
}

// TODO(zaixing.wang): fp16 mark
Tensor& MaskedFill(Tensor& self, const Tensor& mask, const Scalar& value) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::masked_fill__cuda(self, mask, value);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("fill_.Scalar", &Fill);
  m.impl("zero_", &Zero_);
  m.impl("masked_fill_.Scalar", &MaskedFill);
}

} // namespace musa
} // namespace at
