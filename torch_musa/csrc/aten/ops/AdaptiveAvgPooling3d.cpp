#include <ATen/DeviceGuard.h>
#include <ATen/core/Tensor.h>
#include <c10/core/SymIntArrayRef.h>

// #include <ATen/ops/_adaptive_avg_pool3d_backward_native.h>
// #include <ATen/ops/adaptive_avg_pool3d_backward_native.h>

namespace at {
namespace musa {

Tensor& AdaptiveAvgPool3DOutMUSA(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output);

Tensor AdaptiveAvgPool3DMUSA(const Tensor& input, IntArrayRef output_size);

Tensor& AdaptiveAvgPool3DBackwardOutMUSA(
    const Tensor& grad_output,
    const Tensor& input,
    Tensor& grad_input);

Tensor AdaptiveAvgPool3DBackwardMUSA(
    const Tensor& grad_output,
    const Tensor& input);

Tensor& AdaptiveAvgPool3dOut(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::musa::AdaptiveAvgPool3DOutMUSA(input, output_size, output);
}

Tensor AdaptiveAvgPool3d(const Tensor& input, IntArrayRef output_size) {
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::musa::AdaptiveAvgPool3DMUSA(input, output_size);
}

Tensor& AdaptiveAvgPool3dBackwardOut(
    const Tensor& grad_output,
    const Tensor& self,
    Tensor& grad_input) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::musa::AdaptiveAvgPool3DBackwardOutMUSA(
      grad_output, self, grad_input);
}

Tensor AdaptiveAvgPool3dBackward(
    const Tensor& grad_output,
    const Tensor& self) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::musa::AdaptiveAvgPool3DBackwardMUSA(grad_output, self);
}

} // namespace musa
} // namespace at
