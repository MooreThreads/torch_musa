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
          input.scalar_type() == at::ScalarType::Half,
      "Dtype of input tensor of LayerNorm only support Float32,Half, but now it is ",
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
  muTensor mt_gamma;
  muTensor mt_beta;
  Tensor gamma;
  if (weight.defined()) {
    gamma = weight.contiguous();
    mt_gamma = CreateMUTensor(gamma);
    TORCH_CHECK(
        weight.scalar_type() == at::ScalarType::Float ||
            weight.scalar_type() == at::ScalarType::Half,
        "Dtype of weight tensor of LayerNorm only support Float32, Half",
        "but now it is ",
        weight.scalar_type());
  }
  if (bias.defined()) {
    auto beta = bias.contiguous();
    mt_beta = CreateMUTensor(beta);
    TORCH_CHECK(
        bias.scalar_type() == at::ScalarType::Float ||
            bias.scalar_type() == at::ScalarType::Half,
        "Dtype of bias tensor of LayerNorm only support Float32, Half",
        "but now it is ",
        bias.scalar_type());
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
      op.Run(h, mt_output, mt_mean, mt_rstd, mt_input, mt_gamma, mt_beta),
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
          grad_out.scalar_type() == at::ScalarType::Half,
      "Dtype of grad_out tensor of LayerNormBackward only support Float32/Half, ",
      "but now it is ",
      grad_out.scalar_type());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float ||
          input.scalar_type() == at::ScalarType::Half,
      "Dtype of input tensor of LayerNormBackward only support Float32/Half, ",
      "but now it is ",
      input.scalar_type());
  TORCH_CHECK(
      mean.scalar_type() == at::ScalarType::Float ||
          mean.scalar_type() == at::ScalarType::Half,
      "Dtype of mean tensor of LayerNormBackward only support Float32/Half, ",
      "but now it is ",
      mean.scalar_type());
  TORCH_CHECK(
      rstd.scalar_type() == at::ScalarType::Float ||
          rstd.scalar_type() == at::ScalarType::Half,
      "Dtype of rstd tensor of LayerNormBackward only support Float32/Half, ",
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
              weight.scalar_type() == at::ScalarType::Half,
          "Dtype of weight tensor of LayerNormBackward only support Float32/Half, ",
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

at::Tensor RMSNorm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight_opt /* optional */,
    double eps) {
  // Check
  const int normalized_ndim = normalized_shape.size();
  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of RMSNorm must be MUSA, but now is ",
      input.device());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float ||
          input.scalar_type() == at::ScalarType::Half,
      "Dtype of input tensor of RMSNorm only support Float32,Half, but now it is ",
      input.scalar_type());
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() || weight.sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight.sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight.defined() ||
          (weight.scalar_type() == at::ScalarType::Float ||
           weight.scalar_type() == at::ScalarType::Half),
      "Dtype of weight tensor of LayerNorm only support Float32, Half",
      "but now it is ",
      weight.scalar_type());

  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }
  // Device guard
  c10::musa::MUSAGuard device_guard(input.device());
  // Generate ouput && square
  Tensor contiguous_input = input.contiguous();
  auto output = at::empty_like(contiguous_input);

  const int axis = input_ndim - normalized_ndim;
  const int64_t M =
      c10::multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);

  at::TensorOptions options = input.options();
  auto square = at::empty({M}, options);
  if (M > 0) {
    std::vector<int64_t> stat_shape;
    for (int32_t idx = 0; idx < axis; ++idx) {
      stat_shape.push_back(input_shape[idx]);
    }
    for (int32_t idx = axis; idx < input.dim(); ++idx) {
      stat_shape.push_back(1);
    }
    square = square.view(stat_shape);
  }

  // Prepare mudnn input
  auto mt_input = CreateMUTensor(contiguous_input);
  auto mt_output = CreateMUTensor(output);
  auto mt_square = CreateMUTensor(square);
  muTensor mt_gamma;
  Tensor gamma;
  if (weight.defined()) {
    gamma = weight.contiguous();
    mt_gamma = CreateMUTensor(gamma);
  }

  std::vector<int32_t> norm_axis;
  const int32_t diff = input.dim() - normalized_shape.size();
  for (size_t i = 0; i != normalized_shape.size(); ++i) {
    if (input.size(diff + i) == normalized_shape[i]) {
      norm_axis.push_back(diff + i);
    }
  }

  // Call mudnn
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::RMSNorm op;
  CHECK_MUDNN_STATUS(op.SetAxis(norm_axis.size(), norm_axis.data()), "SetAxis");
  CHECK_MUDNN_STATUS(op.SetEpsilon(eps), "SetEpsilon");
  CHECK_MUDNN_STATUS(
      op.Run(h, mt_output, mt_square, mt_input, mt_gamma), "Run");
  return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("native_layer_norm", &NativeLayerNorm);
  m.impl("native_layer_norm_backward", &NativeLayerNormBwd);
  m.impl("rms_norm", &RMSNorm);
}

} // namespace musa
} // namespace at
