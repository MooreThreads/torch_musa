#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>
#include <ATen/native/UpSample.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

at::Tensor& GradInputSmoothL1LossBackwardOut(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta,
    at::Tensor& grad_input) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device,
      grad_input,
      "grad_input_smooth_l1_loss_backward_out",
      "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "grad_input_smooth_l1_loss_backward_out",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, self, "grad_input_smooth_l1_loss_backward_out", "self");
  c10::impl::check_and_update_common_device(
      common_device,
      target,
      "grad_input_smooth_l1_loss_backward_out",
      "target");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::smooth_l1_loss_backward_out(
      grad_output, self, target, reduction, beta, grad_input);
}

} // namespace musa
} // namespace at
