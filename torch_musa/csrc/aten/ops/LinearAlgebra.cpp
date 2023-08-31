#include <ATen/Config.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/linalg_lstsq_native.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> LinalgLstsqOut(
    const at::Tensor& self,
    const at::Tensor& b,
    c10::optional<double> rcond,
    c10::optional<c10::string_view> driver,
    at::Tensor& solution,
    at::Tensor& residuals,
    at::Tensor& rank,
    at::Tensor& singular_values) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, solution, "LinalgLstsqOut", "solution");
  c10::impl::check_and_update_common_device(
      common_device, residuals, "LinalgLstsqOut", "residuals");
  c10::impl::check_and_update_common_device(
      common_device, rank, "LinalgLstsqOut", "rank");
  c10::impl::check_and_update_common_device(
      common_device, singular_values, "LinalgLstsqOut", "singular_values");
  c10::impl::check_and_update_common_device(
      common_device, self, "LinalgLstsqOut", "self");
  c10::impl::check_and_update_common_device(
      common_device, b, "LinalgLstsqOut", "b");
  const OptionalDeviceGuard device_guard(device_of(self));

  auto cpu_self =
      at::empty(self.sizes(), self.options().device(DeviceType::CPU));
  auto cpu_b = at::empty(b.sizes(), b.options().device(DeviceType::CPU));
  auto cpu_solution =
      at::empty(solution.sizes(), solution.options().device(DeviceType::CPU));
  auto cpu_residuals =
      at::empty(residuals.sizes(), residuals.options().device(DeviceType::CPU));
  auto cpu_rank =
      at::empty(rank.sizes(), rank.options().device(DeviceType::CPU));
  auto cpu_singular_values = at::empty(
      singular_values.sizes(),
      singular_values.options().device(DeviceType::CPU));

  cpu_self.copy_(self);
  cpu_b.copy_(b);
  cpu_solution.copy_(solution);
  cpu_residuals.copy_(residuals);
  cpu_rank.copy_(rank);
  cpu_singular_values.copy_(singular_values);

  at::native::linalg_lstsq_out(
      cpu_self,
      cpu_b,
      rcond,
      driver,
      cpu_solution,
      cpu_residuals,
      cpu_rank,
      cpu_singular_values);

  solution = at::empty(
      cpu_solution.sizes(),
      cpu_solution.options().device(DeviceType::PrivateUse1));
  solution.copy_(cpu_solution);
  residuals = at::empty(
      cpu_residuals.sizes(),
      cpu_residuals.options().device(DeviceType::PrivateUse1));
  residuals.copy_(cpu_residuals);
  rank = at::empty(
      cpu_rank.sizes(), cpu_rank.options().device(DeviceType::PrivateUse1));
  rank.copy_(cpu_rank);
  singular_values = at::empty(
      cpu_singular_values.sizes(),
      cpu_singular_values.options().device(DeviceType::PrivateUse1));
  singular_values.copy_(cpu_singular_values);

  return ::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>{
      solution, residuals, rank, singular_values};
  ;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // It requires mublas geqrfBatched function which is not implemented, so use
  // cpu to make workaround
  m.impl("linalg_lstsq.out", &LinalgLstsqOut);
}
} // namespace musa
} // namespace at