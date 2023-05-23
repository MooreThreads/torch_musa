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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("flip", &Flip);
}

} // namespace musa
} // namespace at
