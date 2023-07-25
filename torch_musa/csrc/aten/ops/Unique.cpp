#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

::std::tuple<at::Tensor, at::Tensor, at::Tensor> Unique2(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "Unique2", "self");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::_unique2_cuda(self, sorted, return_inverse, return_counts);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_unique2", &Unique2);
}

} // namespace musa
} // namespace at