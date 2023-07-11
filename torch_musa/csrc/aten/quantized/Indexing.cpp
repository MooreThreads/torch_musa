#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ops/index_select_native.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace native {

Tensor& IndexSelectOut(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::index_select_out_cuda(self, dim, index, out);
}

Tensor IndexSelect(const Tensor& self, int64_t dim, const Tensor& index) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::index_select_quantized_cuda(self, dim, index);
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("index_select.out", &IndexSelectOut);
  m.impl("index_select", &IndexSelect);
}

} // namespace native
} // namespace at
