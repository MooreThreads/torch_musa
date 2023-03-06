#include <ATen/Config.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/nll_loss_backward_native.h>
#include <ATen/ops/nll_loss_forward_native.h>
#endif

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

#include <mudnn.h>

namespace at {
namespace musa {
namespace {
struct structured_nll_loss_backward_out_cuda_functional final
    : public at::native::structured_nll_loss_backward_out_cuda {
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
  }
  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }
  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_nll_loss_forward_out_cuda_functional final
    : public at::native::structured_nll_loss_forward_out_cuda {
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
  }
  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }
  std::array<c10::ExclusivelyOwned<Tensor>, 2> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_nll_loss_forward_out_cuda_out final
    : public at::native::structured_nll_loss_forward_out_cuda {
  structured_nll_loss_forward_out_cuda_out(Tensor& out0, Tensor& out1)
      : outputs_{std::ref(out0), std::ref(out1)} {}
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
    if (C10_UNLIKELY(maybe_proxy.has_value())) {
      proxy_outputs_[output_idx] =
          c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());
    }
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
  }
  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 2> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 2> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_nll_loss_backward_out_cuda_out final
    : public at::native::structured_nll_loss_backward_out_cuda {
  structured_nll_loss_backward_out_cuda_out(Tensor& out0)
      : outputs_{std::ref(out0)} {}
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
    if (C10_UNLIKELY(maybe_proxy.has_value())) {
      proxy_outputs_[output_idx] =
          c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());
    }
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
  }
  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

} // namespace
::std::tuple<at::Tensor&, at::Tensor&> wrapper_CUDA_nll_loss_forward_out_output(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    at::Tensor& output,
    at::Tensor& total_weight) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device,
      output,
      "wrapper_CUDA_nll_loss_forward_out_output",
      "output");
  c10::impl::check_and_update_common_device(
      common_device,
      total_weight,
      "wrapper_CUDA_nll_loss_forward_out_output",
      "total_weight");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_CUDA_nll_loss_forward_out_output", "self");
  c10::impl::check_and_update_common_device(
      common_device,
      target,
      "wrapper_CUDA_nll_loss_forward_out_output",
      "target");
  c10::impl::check_and_update_common_device(
      common_device,
      weight,
      "wrapper_CUDA_nll_loss_forward_out_output",
      "weight");
  structured_nll_loss_forward_out_cuda_out op(output, total_weight);
  op.meta(
      self,
      target,
      ((weight.has_value() && (*weight).defined())
           ? at::OptionalTensorRef(*weight)
           : at::OptionalTensorRef()),
      reduction,
      ignore_index);
  op.impl(
      self,
      target,
      ((weight.has_value() && (*weight).defined())
           ? at::OptionalTensorRef(*weight)
           : at::OptionalTensorRef()),
      reduction,
      ignore_index,
      op.maybe_get_output(0),
      op.maybe_get_output(1));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  if (op.proxy_outputs_[1].has_value())
    op.outputs_[1].get().copy_(**op.proxy_outputs_[1]);
  return std::forward_as_tuple(output, total_weight);
}

::std::tuple<at::Tensor, at::Tensor> wrapper_CUDA_nll_loss_forward(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_CUDA_nll_loss_forward", "self");
  c10::impl::check_and_update_common_device(
      common_device, target, "wrapper_CUDA_nll_loss_forward", "target");
  c10::impl::check_and_update_common_device(
      common_device, weight, "wrapper_CUDA_nll_loss_forward", "weight");
  structured_nll_loss_forward_out_cuda_functional op;
  op.meta(
      self,
      target,
      ((weight.has_value() && (*weight).defined())
           ? at::OptionalTensorRef(*weight)
           : at::OptionalTensorRef()),
      reduction,
      ignore_index);
  op.impl(
      self,
      target,
      ((weight.has_value() && (*weight).defined())
           ? at::OptionalTensorRef(*weight)
           : at::OptionalTensorRef()),
      reduction,
      ignore_index,
      *op.outputs_[0],
      *op.outputs_[1]);
  return std::make_tuple(
      std::move(op.outputs_[0]).take(), std::move(op.outputs_[1]).take());
}

at::Tensor wrapper_CUDA_nll_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "wrapper_CUDA_nll_loss_backward",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_CUDA_nll_loss_backward", "self");
  c10::impl::check_and_update_common_device(
      common_device, target, "wrapper_CUDA_nll_loss_backward", "target");
  c10::impl::check_and_update_common_device(
      common_device, weight, "wrapper_CUDA_nll_loss_backward", "weight");
  c10::impl::check_and_update_common_device(
      common_device,
      total_weight,
      "wrapper_CUDA_nll_loss_backward",
      "total_weight");
  structured_nll_loss_backward_out_cuda_functional op;
  op.meta(
      grad_output,
      self,
      target,
      ((weight.has_value() && (*weight).defined())
           ? at::OptionalTensorRef(*weight)
           : at::OptionalTensorRef()),
      reduction,
      ignore_index,
      total_weight);
  op.impl(
      grad_output,
      self,
      target,
      ((weight.has_value() && (*weight).defined())
           ? at::OptionalTensorRef(*weight)
           : at::OptionalTensorRef()),
      reduction,
      ignore_index,
      total_weight,
      *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

at::Tensor& wrapper_CUDA_nll_loss_backward_out_grad_input(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight,
    at::Tensor& grad_input) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device,
      grad_input,
      "wrapper_CUDA_nll_loss_backward_out_grad_input",
      "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "wrapper_CUDA_nll_loss_backward_out_grad_input",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device,
      self,
      "wrapper_CUDA_nll_loss_backward_out_grad_input",
      "self");
  c10::impl::check_and_update_common_device(
      common_device,
      target,
      "wrapper_CUDA_nll_loss_backward_out_grad_input",
      "target");
  c10::impl::check_and_update_common_device(
      common_device,
      weight,
      "wrapper_CUDA_nll_loss_backward_out_grad_input",
      "weight");
  c10::impl::check_and_update_common_device(
      common_device,
      total_weight,
      "wrapper_CUDA_nll_loss_backward_out_grad_input",
      "total_weight");
  structured_nll_loss_backward_out_cuda_out op(grad_input);
  op.meta(
      grad_output,
      self,
      target,
      ((weight.has_value() && (*weight).defined())
           ? at::OptionalTensorRef(*weight)
           : at::OptionalTensorRef()),
      reduction,
      ignore_index,
      total_weight);
  op.impl(
      grad_output,
      self,
      target,
      ((weight.has_value() && (*weight).defined())
           ? at::OptionalTensorRef(*weight)
           : at::OptionalTensorRef()),
      reduction,
      ignore_index,
      total_weight,
      op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return grad_input;
}

extern Tensor create_out(
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options);
extern void resize_out(
    const Tensor& out,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options);

ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "nll_loss_forward.output",
    wrapper_CUDA_nll_loss_forward_out_output)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "nll_loss_forward",
    wrapper_CUDA_nll_loss_forward)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "nll_loss_backward.grad_input",
    wrapper_CUDA_nll_loss_backward_out_grad_input)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "nll_loss_backward",
    wrapper_CUDA_nll_loss_backward)

} // namespace musa
} // namespace at
