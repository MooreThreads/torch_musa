#include <ATen/Config.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {
extern Tensor create_out(
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options);
extern void resize_out(
    const Tensor& out,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options);

std::tuple<at::Tensor&, at::Tensor&> NllLossOut(
    const at::Tensor& input,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    at::Tensor& output,
    at::Tensor& total_weight) {
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of NllLoss must be MUSA, but now is ",
      input.device());
  TORCH_CHECK(
      target.device().type() == kMUSA,
      "Device of target tensor of NllLoss must be MUSA, but now is ",
      target.device());
  TORCH_CHECK(
      output.device().type() == kMUSA,
      "Device of output tensor of NllLoss must be MUSA, but now is ",
      output.device());
  TORCH_CHECK(
      total_weight.device().type() == kMUSA,
      "Device of total_weight tensor of NllLoss must be MUSA, but now is ",
      total_weight.device());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of NllLoss only support Float32, but now it is ",
      input.scalar_type());
  c10::musa::MUSAGuard device_guard(input.device());

  auto contiguous_input = Contiguous(input);
  auto contiguous_target = Contiguous(target);
  TORCH_CHECK(
      input.dim() > 0 && input.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() <= 1,
      "0D or 1D target tensor expected, multi-target not supported");
  auto no_batch_dim = input.dim() == 1 && target.dim() == 0;
  TORCH_CHECK(
      no_batch_dim || (input.size(0) == target.size(0)),
      "size mismatch (got input: ",
      input.sizes(),
      ", target: ",
      target.sizes(),
      ")")

  const auto n_classes = input.size(-1);
  bool has_weight = false;
  if (weight.has_value() && (*weight).defined()) {
    has_weight = true;
    TORCH_CHECK(
        weight.value().device().type() == kMUSA,
        "Device of weight tensor of NllLoss must be MUSA, but now is ",
        weight.value().device());
    TORCH_CHECK(
        weight.value().scalar_type() == at::ScalarType::Float,
        "Dtype of weight tensor of NllLoss only support Float32, ",
        "but now it is ",
        weight.value().scalar_type());
  }
  TORCH_CHECK(
      !has_weight ||
          (weight.value().dim() == 1 && weight.value().numel() == n_classes),
      "weight tensor should be defined either for all ",
      n_classes,
      " classes or no classes"
      " but got weight tensor of shape: ",
      weight.value().sizes());

  const auto n_dims = input.dim();
  const auto batch_size = input.size(0);
  if (reduction == Reduction::None && n_dims == 2) {
    output.resize_({batch_size});
  } else {
    // produce scalar output when reducing or input is 1d
    output.resize_({});
  }
  total_weight.resize_({});

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::NLLLoss nll_loss_op;
  auto mt_input = CreateMUTensor(contiguous_input);
  auto mt_target = CreateMUTensor(contiguous_target);
  auto mt_output = CreateMUTensor(output);
  auto mt_total_weight = CreateMUTensor(total_weight);
  muTensor mt_weight;
  if (has_weight) {
    auto contiguous_weight = Contiguous(weight.value());
    mt_weight = CreateMUTensor(contiguous_weight);
  }
  CHECK_MUDNN_STATUS(
      nll_loss_op.SetReductionMode(
          static_cast<::musa::dnn::NLLLoss::Mode>(reduction)),
      "SetReductionMode");
  CHECK_MUDNN_STATUS(
      nll_loss_op.SetIgnoreIndex(ignore_index), "SetIgnoreIndex");
  CHECK_MUDNN_STATUS(
      nll_loss_op.Run(
          h,
          mt_output,
          mt_total_weight,
          mt_input,
          mt_target,
          mt_weight,
          InternalMemAlloc),
      "Run");
  return std::forward_as_tuple(output, total_weight);
}

std::tuple<at::Tensor, at::Tensor> NllLoss(
    const at::Tensor& input,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index) {
  auto output = at::empty({0}, input.options());
  auto total_weight = at::empty(input.sizes(), input.options());
  NllLossOut(
      input, target, weight, reduction, ignore_index, output, total_weight);
  return std::forward_as_tuple(output, total_weight);
}

at::Tensor& NllLossBwdGradInput(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight,
    at::Tensor& grad_input) {
  TORCH_CHECK(
      grad_output.device().type() == kMUSA,
      "Device of grad_output tensor of NllLossBackward must be MUSA, ",
      "but now is ",
      grad_output.device());
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of NllLossBackward must be MUSA, but now is ",
      input.device());
  TORCH_CHECK(
      target.device().type() == kMUSA,
      "Device of target tensor of NllLossBackward must be MUSA, but now is ",
      target.device());
  TORCH_CHECK(
      total_weight.device().type() == kMUSA,
      "Device of total_weight tensor of NllLossBackward must be MUSA, ",
      "but now is ",
      total_weight.device());
  TORCH_CHECK(
      grad_input.device().type() == kMUSA,
      "Device of grad_input tensor of NllLossBackward must be MUSA, ",
      "but now is ",
      grad_input.device());
  TORCH_CHECK(
      grad_output.scalar_type() == at::ScalarType::Float,
      "Dtype of grad_output tensor of NllLossBackward only support Float32, ",
      "but now it is ",
      grad_output.scalar_type());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of NllLossBackward only support Float32, ",
      "but now it is ",
      input.scalar_type());
  c10::musa::MUSAGuard guard_device(input.device());
  auto contiguous_grad_output = Contiguous(grad_output);
  auto contiguous_input = Contiguous(input);
  auto contiguous_target = Contiguous(target);
  auto contiguous_total_weight = Contiguous(total_weight);

  TORCH_CHECK(
      input.dim() > 0 && input.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() <= 1,
      "0D or 1D target tensor expected, multi-target not supported");
  auto no_batch_dim = input.dim() == 1 && target.dim() == 0;
  TORCH_CHECK(
      no_batch_dim || (input.size(0) == target.size(0)),
      "size mismatch (got input: ",
      input.sizes(),
      ", target: ",
      target.sizes(),
      ")")
  TORCH_CHECK(
      total_weight.numel() == 1,
      "expected total_weight to be a  single element tensor, got: ",
      total_weight.sizes(),
      " (",
      total_weight.numel(),
      " elements)");

  bool has_weight = false;
  if (weight.has_value() && (*weight).defined()) {
    has_weight = true;
    TORCH_CHECK(
        weight.value().device().type() == kMUSA,
        "Device of weight tensor of NllLossBackward must be MUSA, but now is ",
        weight.value().device());
    TORCH_CHECK(
        weight.value().scalar_type() == at::ScalarType::Float,
        "Dtype of weight tensor of NllLossBackward only support Float32, ",
        "but now it is ",
        weight.value().scalar_type());
  }
  TORCH_CHECK(
      !has_weight || weight.value().numel() == input.size(-1),
      "weight tensor should be defined either for all or no classes");

  const auto n_dims = input.dim();
  if (reduction == Reduction::None && n_dims == 2) {
    const auto batch_size = input.size(0);
    check_dim_size(grad_output, 1, 0, batch_size);
  } else {
    TORCH_CHECK(
        grad_output.dim() <= 1 && grad_output.numel() == 1,
        "Expected a single element grad_output tensor, but got: ",
        grad_output.sizes());
  }
  grad_input.resize_(input.sizes());

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::NLLLoss nll_loss_op;
  auto mt_target = CreateMUTensor(contiguous_target);
  auto mt_grad_output = CreateMUTensor(contiguous_grad_output);
  auto mt_total_weight = CreateMUTensor(contiguous_total_weight);
  auto mt_grad_input = CreateMUTensor(grad_input);
  muTensor mt_weight;
  if (has_weight) {
    auto contiguous_weight = Contiguous(weight.value());
    mt_weight = CreateMUTensor(contiguous_weight);
  }
  CHECK_MUDNN_STATUS(
      nll_loss_op.SetReductionMode(
          static_cast<::musa::dnn::NLLLoss::Mode>(reduction)),
      "SetReductionMode");
  CHECK_MUDNN_STATUS(
      nll_loss_op.SetIgnoreIndex(ignore_index), "SetIgnoreIndex");
  CHECK_MUDNN_STATUS(
      nll_loss_op.RunBwd(
          h,
          mt_grad_input,
          mt_grad_output,
          mt_target,
          mt_weight,
          mt_total_weight,
          InternalMemAlloc),
      "Run");
  return grad_input;
}

at::Tensor NllLossBwd(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight) {
  auto grad_input = at::empty(
      input.sizes(),
      input.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  NllLossBwdGradInput(
      grad_output,
      input,
      target,
      weight,
      reduction,
      ignore_index,
      total_weight,
      grad_input);
  return grad_input;
}

Tensor KLDiv(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    bool log_target) {
  c10::musa::MUSAGuard device_guard(input.device());
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::KLDivLoss kldiv;

  Tensor output;
  if (reduction == at::Reduction::Mean) {
    kldiv.SetReductionMode(::musa::dnn::KLDivLoss::Mode::MEAN);
    output = at::empty({}, input.options());
  } else if (reduction == at::Reduction::Sum) {
    kldiv.SetReductionMode(::musa::dnn::KLDivLoss::Mode::SUM);
    output = at::empty({}, input.options());
  } else {
    kldiv.SetReductionMode(::musa::dnn::KLDivLoss::Mode::NONE);
    output = at::empty_like(input);
  }
  kldiv.SetLogTarget(log_target);

  Tensor contiguous_input = Contiguous(input);
  auto mt_input = CreateMUTensor(contiguous_input);
  Tensor contiguous_target = Contiguous(target);
  auto mt_target = CreateMUTensor(contiguous_target);
  auto mt_output = CreateMUTensor(output);

  kldiv.Run(h, mt_output, mt_input, mt_target, InternalMemAlloc);
  return output;
}

Tensor KLDivBwd(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    bool log_target) {
  c10::musa::MUSAGuard device_guard(input.device());
  auto grad_input = at::zeros_like(input);
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::KLDivLoss kldiv;

  if (reduction == at::Reduction::Mean) {
    kldiv.SetReductionMode(::musa::dnn::KLDivLoss::Mode::MEAN);
  } else if (reduction == at::Reduction::Sum) {
    kldiv.SetReductionMode(::musa::dnn::KLDivLoss::Mode::SUM);
  } else {
    kldiv.SetReductionMode(::musa::dnn::KLDivLoss::Mode::NONE);
  }
  kldiv.SetLogTarget(log_target);

  Tensor grad_ = Contiguous(grad);
  auto mt_grad = CreateMUTensor(grad_);
  Tensor contiguous_input = Contiguous(input);
  auto mt_input = CreateMUTensor(contiguous_input);
  Tensor contiguous_target = Contiguous(target);
  auto mt_target = CreateMUTensor(contiguous_target);
  auto mt_gradin = CreateMUTensor(grad_input);

  kldiv.RunBwd(h, mt_gradin, mt_grad, mt_input, mt_target);
  return grad_input;
}

struct structured_mse_loss_out_out final
    : public at::native::structured_mse_loss_out {
  structured_mse_loss_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}

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
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_mse_loss_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
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
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_mse_loss_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

Tensor& MseLossOut(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& out) {
  // No device check

  structured_mse_loss_out_out op(out);
  op.meta(self, target, reduction);
  op.impl(self, target, reduction, op.maybe_get_output(0));
  return out;
}

Tensor MseLossBwd(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, grad_output, "MseLossBwd", "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, self, "MseLossBwd", "self");
  c10::impl::check_and_update_common_device(
      common_device, target, "MseLossBwd", "target");

  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::mse_loss_backward(grad_output, self, target, reduction);
}

at::Tensor& MseLossBwdGradInput(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& grad_input) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, grad_input, "MseLossBwdGradInput", "grad_input");
  c10::impl::check_and_update_common_device(
      common_device, grad_output, "MseLossBwdGradInput", "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, self, "MseLossBwdGradInput", "self");
  c10::impl::check_and_update_common_device(
      common_device, target, "MseLossBwdGradInput", "target");

  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::mse_loss_backward_out(
      grad_output, self, target, reduction, grad_input);
}

struct structured_mse_loss_out_functional final
    : public at::native::structured_mse_loss_out {
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
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_mse_loss_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
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
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_mse_loss_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }
  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

Tensor MseLoss(const Tensor& self, const Tensor& target, int64_t reduction) {
  // No device check

  structured_mse_loss_out_functional op;
  op.meta(self, target, reduction);
  op.impl(self, target, reduction, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mse_loss", &MseLoss);
  m.impl("mse_loss.out", &MseLossOut);
  m.impl("mse_loss_backward", &MseLossBwd);
  m.impl("mse_loss_backward.grad_input", &MseLossBwdGradInput);

  m.impl("nll_loss_forward.output", &NllLossOut);
  m.impl("nll_loss_forward", &NllLoss);
  m.impl("nll_loss_backward.grad_input", &NllLossBwdGradInput);
  m.impl("nll_loss_backward", &NllLossBwd);
  m.impl("kl_div", &KLDiv);
  m.impl("kl_div_backward", &KLDivBwd);
}

} // namespace musa
} // namespace at
