#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

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

ADVANCED_REGISTER(aten, PrivateUse1, "flip", Flip)
ADVANCED_REGISTER(aten, PrivateUse1, "roll", Roll)

REDEFINE_REGISTER(aten, QuantizedPrivateUse1, "flip", Flip)

} // namespace musa
} // namespace at
