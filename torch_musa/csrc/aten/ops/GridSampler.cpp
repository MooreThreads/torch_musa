#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/UpSample.h>

#include <torch/library.h>

#include <mudnn_image.h>
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/musa/GridSampler.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/grid_sampler_2d_backward_native.h>
#include <ATen/ops/grid_sampler_2d_native.h>
#include <ATen/ops/grid_sampler_3d_backward_native.h>
#include <ATen/ops/grid_sampler_3d_native.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::native {

Tensor grid_sampler_2d_cuda(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2]}, input.options());
  launch_grid_sampler_2d_forward_kernel(
      output, input, grid, interpolation_mode, padding_mode, align_corners);
  return output;
}

Tensor grid_sampler_3d_cuda(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]},
      input.options());
  launch_grid_sampler_3d_forward_kernel(
      output, input, grid, interpolation_mode, padding_mode, align_corners);
  return output;
}

std::tuple<Tensor, Tensor> grid_sampler_2d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask) {
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_grid_sampler_2d_backward_kernel(
      grad_input,
      grad_grid,
      grad_output,
      input,
      grid,
      interpolation_mode,
      padding_mode,
      align_corners,
      output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

std::tuple<Tensor, Tensor> grid_sampler_3d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask) {
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_grid_sampler_3d_backward_kernel(
      grad_input,
      grad_grid,
      grad_output,
      input,
      grid,
      interpolation_mode,
      padding_mode,
      align_corners,
      output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

} // namespace at::native

namespace at {
namespace musa {

at::Tensor GridSampler2d(
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, input, "GridSampler2d", "input");
  c10::impl::check_and_update_common_device(
      common_device, grid, "GridSampler2d", "grid");
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::native::grid_sampler_2d_cuda(
      input, grid, interpolation_mode, padding_mode, align_corners);
}

::std::tuple<at::Tensor, at::Tensor> GridSampler2dBackward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    ::std::array<bool, 2> output_mask) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, grad_output, "GridSampler2dBackward", "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, input, "GridSampler2dBackward", "input");
  c10::impl::check_and_update_common_device(
      common_device, grid, "GridSampler2dBackward", "grid");
  const OptionalDeviceGuard device_guard(device_of(grad_output));
  return at::native::grid_sampler_2d_backward_cuda(
      grad_output,
      input,
      grid,
      interpolation_mode,
      padding_mode,
      align_corners,
      output_mask);
}

at::Tensor GridSampler3d(
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, input, "GridSampler3d", "input");
  c10::impl::check_and_update_common_device(
      common_device, grid, "GridSampler3d", "grid");
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::native::grid_sampler_3d_cuda(
      input, grid, interpolation_mode, padding_mode, align_corners);
}

::std::tuple<at::Tensor, at::Tensor> GridSampler3dBackward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    ::std::array<bool, 2> output_mask) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, grad_output, "GridSampler3dBackward", "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, input, "GridSampler3dBackward", "input");
  c10::impl::check_and_update_common_device(
      common_device, grid, "GridSampler3dBackward", "grid");
  const OptionalDeviceGuard device_guard(device_of(grad_output));
  return at::native::grid_sampler_3d_backward_cuda(
      grad_output,
      input,
      grid,
      interpolation_mode,
      padding_mode,
      align_corners,
      output_mask);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("grid_sampler_2d", &GridSampler2d);
  m.impl("grid_sampler_3d", &GridSampler3d);
}

} // namespace musa
} // namespace at
