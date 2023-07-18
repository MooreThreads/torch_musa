#include <ATen/core/op_registration/adaption.h>
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale_native.h>
#include <ATen/ops/_amp_update_scale_native.h>
#include "torch_musa/csrc/aten/ops/TensorFactory.h"

namespace at {
namespace musa {

void AmpForeachNonFiniteCheckAndUnscale(
    at::TensorList self,
    at::Tensor& found_inf,
    const at::Tensor& inv_scale) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device,
      self,
      "wrapper_CUDA___amp_foreach_non_finite_check_and_unscale_",
      "self");
  c10::impl::check_and_update_common_device(
      common_device,
      found_inf,
      "wrapper_CUDA___amp_foreach_non_finite_check_and_unscale_",
      "found_inf");
  c10::impl::check_and_update_common_device(
      common_device,
      inv_scale,
      "wrapper_CUDA___amp_foreach_non_finite_check_and_unscale_",
      "inv_scale");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::_amp_foreach_non_finite_check_and_unscale_cuda_(
      self, found_inf, inv_scale);
}

at::Tensor& AmpUpdateScale(
    at::Tensor& self,
    at::Tensor& growth_tracker,
    const at::Tensor& found_inf,
    double scale_growth_factor,
    double scale_backoff_factor,
    int64_t growth_interval) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_CUDA___amp_update_scale_", "self");
  c10::impl::check_and_update_common_device(
      common_device,
      growth_tracker,
      "wrapper_CUDA___amp_update_scale_",
      "growth_tracker");
  c10::impl::check_and_update_common_device(
      common_device,
      found_inf,
      "wrapper_CUDA___amp_update_scale_",
      "found_inf");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::_amp_update_scale_cuda_(
      self,
      growth_tracker,
      found_inf,
      scale_growth_factor,
      scale_backoff_factor,
      growth_interval);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl(
      "_amp_foreach_non_finite_check_and_unscale_",
      &AmpForeachNonFiniteCheckAndUnscale);
  m.impl("_amp_update_scale_", &AmpUpdateScale);
}

} // namespace musa
} // namespace at
