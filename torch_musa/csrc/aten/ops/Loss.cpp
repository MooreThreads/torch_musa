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
#include <ATen/ops/_fused_cross_entropy_loss_2d_backward_native.h>
#include <ATen/ops/_fused_cross_entropy_loss_2d_forward_native.h>
#include <ATen/ops/binary_cross_entropy_backward_native.h>
#include <ATen/ops/binary_cross_entropy_native.h>
#include <ATen/ops/cross_entropy_loss_2d_choice_native.h>
#include <ATen/ops/mse_loss_backward_native.h>
#include <ATen/ops/mse_loss_native.h>
#include <ATen/ops/nll_loss2d_backward_native.h>
#include <ATen/ops/nll_loss2d_forward_native.h>
#include <ATen/ops/zeros_like.h>
#endif

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
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

  auto contiguous_input = input.contiguous();
  auto contiguous_target = target.contiguous();
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
  auto contiguous_total_weight = total_weight.contiguous();

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::NLLLoss nll_loss_op;
  auto mt_input = CreateMUTensor(contiguous_input);
  auto mt_target = CreateMUTensor(contiguous_target);
  auto mt_output = CreateMUTensor(output);
  auto mt_total_weight = CreateMUTensor(contiguous_total_weight);
  muTensor mt_weight;
  if (has_weight) {
    auto contiguous_weight = weight.value().contiguous();
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
  c10::musa::MUSAGuard guard_device(input.device());
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
  auto contiguous_grad_output = grad_output.contiguous();
  auto contiguous_input = input.contiguous();
  auto contiguous_target = target.contiguous();
  auto contiguous_total_weight = total_weight.contiguous();

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
    auto contiguous_weight = weight.value().contiguous();
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

std::tuple<at::Tensor&, at::Tensor&> NllLoss2dOut(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    at::Tensor& output,
    at::Tensor& total_weight) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::nll_loss2d_forward_out_cuda(
      self, target, weight, reduction, ignore_index, output, total_weight);
}

std::tuple<at::Tensor, at::Tensor> NllLoss2d(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::nll_loss2d_forward_cuda(
      self, target, weight, reduction, ignore_index);
}

at::Tensor& NllLoss2dBwdGradInput(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight,
    at::Tensor& grad_input) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::nll_loss2d_backward_out_cuda(
      grad_output,
      self,
      target,
      weight,
      reduction,
      ignore_index,
      total_weight,
      grad_input);
}

at::Tensor NllLoss2dBwd(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::nll_loss2d_backward_cuda(
      grad_output, self, target, weight, reduction, ignore_index, total_weight);
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
    output = at::empty({0}, input.options());
  } else {
    kldiv.SetReductionMode(::musa::dnn::KLDivLoss::Mode::NONE);
    output = at::empty_like(input, at::MemoryFormat::Contiguous);
  }
  kldiv.SetLogTarget(log_target);
  Tensor contiguous_input = input.contiguous();
  auto mt_input = CreateMUTensor(contiguous_input);
  Tensor contiguous_target = target.contiguous();
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

  Tensor grad_ = grad.contiguous();
  auto mt_grad = CreateMUTensor(grad_);
  Tensor contiguous_input = input.contiguous();
  auto mt_input = CreateMUTensor(contiguous_input);
  Tensor contiguous_target = target.contiguous();
  auto mt_target = CreateMUTensor(contiguous_target);
  auto mt_gradin = CreateMUTensor(grad_input);

  kldiv.RunBwd(h, mt_gradin, mt_grad, mt_input, mt_target);
  return grad_input;
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

at::Tensor BinaryCrossEntropy(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::binary_cross_entropy_cuda(self, target, weight, reduction);
}

at::Tensor& BinaryCrossEntropyOut(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    at::Tensor& out) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::binary_cross_entropy_out_cuda(
      self, target, weight, reduction, out);
}

at::Tensor BinaryCrossEntropyBackward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, grad_output, "BinaryCrossEntropyBackward", "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, self, "BinaryCrossEntropyBackward", "self");
  c10::impl::check_and_update_common_device(
      common_device, target, "BinaryCrossEntropyBackward", "target");
  c10::impl::check_and_update_common_device(
      common_device, weight, "BinaryCrossEntropyBackward", "weight");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::binary_cross_entropy_backward_cuda(
      grad_output, self, target, weight, reduction);
}

at::Tensor& BinaryCrossEntropyBackwardOut(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    at::Tensor& grad_input) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, grad_input, "BinaryCrossEntropyBackwardOut", "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "BinaryCrossEntropyBackwardOut",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, self, "BinaryCrossEntropyBackwardOut", "self");
  c10::impl::check_and_update_common_device(
      common_device, target, "BinaryCrossEntropyBackwardOut", "target");
  c10::impl::check_and_update_common_device(
      common_device, weight, "BinaryCrossEntropyBackwardOut", "weight");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::binary_cross_entropy_backward_out_cuda(
      grad_output, self, target, weight, reduction, grad_input);
}

int64_t CrossEntropyLoss2dChoice(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing) {
  TORCH_CHECK(self.dim() == 2, "CrossEntropyLoss2d now only support 2D input");
  const auto device_prop = at::musa::getDeviceProperties(self.get_device());

  const bool device_support = (device_prop->major >= 3);
  const bool is_valid_target = (target.dim() == 1);
  const bool is_valid_self = at::isFloatingType(self.scalar_type());

  if (device_support && is_valid_target && is_valid_self) {
    return 1;
  }
  return 0;
}

Tensor FusedCrossEntropyLoss2dFwd(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing) {
  TORCH_CHECK(
      self.size(0) == target.size(0),
      "FusedCrossEntropyLoss2dFwd input and target's shapes don't match");
  const auto device_guard = c10::musa::MUSAGuard(self.device());

  at::Tensor wt = weight.value_or(Tensor());
  at::Tensor output = at::empty({}, self.options());
  if (reduction == Reduction::None) {
    output.resize_({self.size(0)});
  }

  muHandle& h = GetMudnnHandle();

  ::musa::dnn::CrossEntropyLoss loss;
  ::musa::dnn::CrossEntropyLoss::ReductionMode mode;
  switch (reduction) {
    case Reduction::Mean:
      mode = ::musa::dnn::CrossEntropyLoss::ReductionMode::MEAN;
      break;
    case Reduction::Sum:
      mode = ::musa::dnn::CrossEntropyLoss::ReductionMode::SUM;
      break;
    default:
      mode = ::musa::dnn::CrossEntropyLoss::ReductionMode::NONE;
      break;
  }

  auto proxy_self = self.expect_contiguous(MemoryFormat::Contiguous);
  auto proxy_target = target.expect_contiguous(MemoryFormat::Contiguous);
  muTensor in = CreateMUTensor(*proxy_self);
  muTensor lab = CreateMUTensor(*proxy_target);
  muTensor w = wt.numel() > 0 ? CreateMUTensor(wt) : muTensor();
  muTensor out = CreateMUTensor(output);
  CHECK_MUDNN_STATUS(loss.SetReductionMode(mode), "SetReductionMode");
  CHECK_MUDNN_STATUS(
      loss.SetLabelSmoothing(label_smoothing), "SetLabelSmoothing");
  CHECK_MUDNN_STATUS(loss.SetIgnoreIndex(ignore_index), "SetIgnoreIndex");
  CHECK_MUDNN_STATUS(loss.Run(h, out, in, lab, w, InternalMemAlloc), "Run");

  return output;
}

Tensor FusedCrossEntropyLoss2dBwd(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing) {
  const auto device_guard = c10::musa::MUSAGuard(self.device());

  at::Tensor wt = weight.value_or(Tensor());
  at::Tensor grad_input = at::empty_like(
      self, self.options().memory_format(at::MemoryFormat::Contiguous));

  muHandle& h = GetMudnnHandle();

  ::musa::dnn::CrossEntropyLoss loss;
  ::musa::dnn::CrossEntropyLoss::ReductionMode mode;
  switch (reduction) {
    case Reduction::Mean:
      mode = ::musa::dnn::CrossEntropyLoss::ReductionMode::MEAN;
      TORCH_CHECK(
          grad_output.numel() == 1,
          "CrossEntropyBwd with reduction mean requires grad_output.numel() == 1");
      break;
    case Reduction::Sum:
      mode = ::musa::dnn::CrossEntropyLoss::ReductionMode::SUM;
      TORCH_CHECK(
          grad_output.numel() == 1,
          "CrossEntropyBwd with reduction sum requires grad_output.numel() == 1");
      break;
    default:
      mode = ::musa::dnn::CrossEntropyLoss::ReductionMode::NONE;
      break;
  }

  muTensor grad = CreateMUTensor(grad_output.contiguous());
  muTensor in = CreateMUTensor(self.contiguous());
  muTensor lab = CreateMUTensor(target.contiguous());
  muTensor w = wt.numel() > 0 ? CreateMUTensor(wt) : muTensor();
  muTensor grad_in = CreateMUTensor(grad_input);
  CHECK_MUDNN_STATUS(loss.SetReductionMode(mode), "SetReductionMode");
  CHECK_MUDNN_STATUS(
      loss.SetLabelSmoothing(label_smoothing), "SetLabelSmoothing");
  CHECK_MUDNN_STATUS(loss.SetIgnoreIndex(ignore_index), "SetIgnoreIndex");
  CHECK_MUDNN_STATUS(
      loss.RunBwd(h, grad_in, in, grad, lab, w, InternalMemAlloc), "RunBwd");

  return grad_input;
}

} // namespace musa
} // namespace at
