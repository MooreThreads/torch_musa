#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/layer_norm.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

::std::tuple<at::Tensor, at::Tensor, at::Tensor> NativeLayerNorm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */,
    double eps) {
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float ||
          input.scalar_type() == at::ScalarType::Half ||
          input.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of input tensor of LayerNorm only support Float32, Half and BFloat16 but now it is ",
      input.scalar_type());

  c10::musa::MUSAGuard device_guard(input.device());
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = at::native::_check_layer_norm_inputs(
      input, normalized_shape, weight, bias);
  auto M = M_N.first;
  Tensor contiguous_input = input.contiguous();
  auto output = at::empty_like(contiguous_input);

  muHandle& h = GetMudnnHandle();
  auto mt_input = CreateMUTensor(contiguous_input);
  auto mt_output = CreateMUTensor(output);
  muTensor mt_gamma, mt_beta;
  Tensor gamma, beta;
  if (weight.defined()) {
    gamma = weight.contiguous();
    mt_gamma = CreateMUTensor(gamma);
    TORCH_CHECK(
        weight.scalar_type() == at::ScalarType::Float ||
            weight.scalar_type() == at::ScalarType::Half ||
            weight.scalar_type() == at::ScalarType::BFloat16,
        "Dtype of weight tensor of LayerNorm only support Float32, Half and BFloat16",
        "but now it is ",
        weight.scalar_type());
  }
  if (bias.defined()) {
    beta = bias.contiguous();
    mt_beta = CreateMUTensor(beta);
    TORCH_CHECK(
        bias.scalar_type() == at::ScalarType::Float ||
            bias.scalar_type() == at::ScalarType::Half ||
            bias.scalar_type() == at::ScalarType::BFloat16,
        "Dtype of bias tensor of LayerNorm only support Float32, Half and BFloat16",
        "but now it is ",
        bias.scalar_type());
  } else if (weight.defined()) {
    // TODO(MTAI): if weight != None && bias == None
    // mudnn not support in rc3.0.1, so initialize bias to zeros temporarily.
    beta = at::zeros_like(weight);
    mt_beta = CreateMUTensor(beta);
  }

  std::vector<int32_t> norm_axis;
  const int32_t diff = input.dim() - normalized_shape.size();
  for (size_t i = 0; i != normalized_shape.size(); ++i) {
    if (input.size(diff + i) == normalized_shape[i]) {
      norm_axis.push_back(diff + i);
    }
  }
  ::musa::dnn::LayerNorm op;
  CHECK_MUDNN_STATUS(op.SetAxis(norm_axis.size(), norm_axis.data()), "SetAxis");
  CHECK_MUDNN_STATUS(op.SetEpsilon(eps), "SetEpsilon");

  const auto input_shape = input.sizes();
  const int32_t axis = input.dim() - normalized_shape.size();
  at::TensorOptions options = input.options();
  auto mean = at::empty({M}, options);
  auto rstd = at::empty({M}, options);
  if (M > 0) {
    std::vector<int64_t> stat_shape;
    for (int32_t idx = 0; idx < axis; ++idx) {
      stat_shape.push_back(input_shape[idx]);
    }
    for (int32_t idx = axis; idx < input.dim(); ++idx) {
      stat_shape.push_back(1);
    }
    mean = mean.view(stat_shape);
    rstd = rstd.view(stat_shape);
  }
  auto mt_mean = CreateMUTensor(mean);
  auto mt_rstd = CreateMUTensor(rstd);

  CHECK_MUDNN_STATUS(
      op.Run(
          h,
          mt_output,
          mt_mean,
          mt_rstd,
          mt_input,
          mt_gamma,
          mt_beta,
          InternalMemAlloc),
      "Run");
  return std::make_tuple(std::move(output), std::move(mean), std::move(rstd));
}

::std::tuple<Tensor, Tensor, Tensor> NativeLayerNormBwd(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    ::std::array<bool, 3> grad_input_mask) {
  TORCH_CHECK(
      grad_out.scalar_type() == at::ScalarType::Float ||
          grad_out.scalar_type() == at::ScalarType::Half ||
          grad_out.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of grad_out tensor of LayerNormBackward only support Float32/Half/BFloat16, ",
      "but now it is ",
      grad_out.scalar_type());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float ||
          input.scalar_type() == at::ScalarType::Half ||
          input.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of input tensor of LayerNormBackward only support Float32/Half/BFloat16, ",
      "but now it is ",
      input.scalar_type());
  TORCH_CHECK(
      mean.scalar_type() == at::ScalarType::Float ||
          mean.scalar_type() == at::ScalarType::Half ||
          mean.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of mean tensor of LayerNormBackward only support Float32/Half/BFloat16, ",
      "but now it is ",
      mean.scalar_type());
  TORCH_CHECK(
      rstd.scalar_type() == at::ScalarType::Float ||
          rstd.scalar_type() == at::ScalarType::Half ||
          rstd.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of rstd tensor of LayerNormBackward only support Float32/Half/BFloat16, ",
      "but now it is ",
      rstd.scalar_type());
  c10::musa::MUSAGuard device_guard(input.device());
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = at::native::_check_layer_norm_inputs(
      input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto X = input.contiguous();
  auto gamma = weight.defined() ? weight.contiguous() : weight;
  auto beta = bias.defined() ? bias.contiguous() : bias;

  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  muTensor mt_dX;
  muTensor mt_dgamma;
  muTensor mt_dbeta;
  if (grad_input_mask[0]) {
    dX = at::native::empty_like(
        X,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    mt_dX = CreateMUTensor(dX);
  }
  if (grad_input_mask[1]) {
    dgamma = M > 0 ? at::native::empty_like(
                         gamma,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         c10::nullopt /* device */,
                         c10::nullopt /* pin_memory */,
                         LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                   : at::native::zeros_like(
                         gamma,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         c10::nullopt /* device */,
                         c10::nullopt /* pin_memory */,
                         LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    mt_dgamma = CreateMUTensor(dgamma);
  }
  if (grad_input_mask[2]) {
    dbeta = M > 0 ? at::native::empty_like(
                        beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                  : at::native::zeros_like(
                        beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    mt_dbeta = CreateMUTensor(dbeta);
  }
  if (M > 0) {
    auto contiguous_grad_out = grad_out.contiguous();
    auto contiguous_mean = mean.contiguous();
    auto contiguous_rstd = rstd.contiguous();
    auto mt_grad_out = CreateMUTensor(contiguous_grad_out);
    auto mt_X = CreateMUTensor(X);
    auto mt_mean = CreateMUTensor(contiguous_mean);
    auto mt_rstd = CreateMUTensor(contiguous_rstd);
    muTensor mt_weight;
    if (weight.defined()) {
      mt_weight = CreateMUTensor(gamma);
      TORCH_CHECK(
          weight.scalar_type() == at::ScalarType::Float ||
              weight.scalar_type() == at::ScalarType::Half ||
              weight.scalar_type() == at::ScalarType::BFloat16,
          "Dtype of weight tensor of LayerNormBackward only support Float32/Half/BFloat16, ",
          "but now it is ",
          weight.scalar_type());
    }
    muHandle& h = GetMudnnHandle();
    ::musa::dnn::LayerNorm op;
    std::vector<int32_t> norm_axis;
    const int32_t diff = input.dim() - normalized_shape.size();
    for (size_t i = 0; i != normalized_shape.size(); ++i) {
      if (input.size(diff + i) == normalized_shape[i]) {
        norm_axis.push_back(diff + i);
      }
    }
    CHECK_MUDNN_STATUS(
        op.SetAxis(norm_axis.size(), norm_axis.data()), "SetAxis");
    CHECK_MUDNN_STATUS(
        op.RunBwd(
            h,
            mt_dX,
            mt_dgamma,
            mt_dbeta,
            mt_grad_out,
            mt_X,
            mt_mean,
            mt_rstd,
            mt_weight,
            InternalMemAlloc),
        "Run");
  }
  return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
}

} // namespace musa
} // namespace at
