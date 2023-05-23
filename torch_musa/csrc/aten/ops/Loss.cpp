#include <ATen/Config.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("nll_loss_forward.output", &NllLossOut);
  m.impl("nll_loss_forward", &NllLoss);
  m.impl("nll_loss_backward.grad_input", &NllLossBwdGradInput);
  m.impl("nll_loss_backward", &NllLossBwd);
  m.impl("kl_div", &KLDiv);
  m.impl("kl_div_backward", &KLDivBwd);
}

} // namespace musa
} // namespace at
