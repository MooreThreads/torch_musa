#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

at::Tensor Flip(const at::Tensor& self, at::IntArrayRef dims) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::flip(self, dims);
}

at::Tensor Roll(
    const at::Tensor& self,
    at::IntArrayRef shifts,
    at::IntArrayRef dims) {
  c10::musa::MUSAGuard device_guard(self.device());
  // TODO(@zhi-cai): remove cuda strings that may appear during cuda-porting
  return at::native::roll_cuda(self, shifts, dims);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("flip", &Flip);
  m.impl("roll", &Roll);
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("flip", &Flip);
}

} // namespace musa
} // namespace at
