#include <ATen/AccumulateType.h>
#include <ATen/Config.h>
#include <ATen/native/layer_norm.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/native_layer_norm_backward_native.h>
#include <ATen/ops/native_layer_norm_native.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/zeros_like_native.h>
#endif

#include <iostream>
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
namespace at::musa {

using Proxy = c10::MaybeOwned<Tensor>;

namespace {
void MaybeSetVarMode(const Tensor& input, ::musa::dnn::LayerNorm& op) {
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3100)
  if (input.dtype() == at::ScalarType::Float) {
    CHECK_MUDNN_STATUS(
        op.SetVarMode(::musa::dnn::LayerNorm::VarMode::WELFORD), "SetMode");
  } else {
    CHECK_MUDNN_STATUS(
        op.SetVarMode(::musa::dnn::LayerNorm::VarMode::DIRECT), "SetMode");
  }
#endif
}
} // anonymous namespace

std::tuple<Tensor, Tensor, Tensor> NativeLayerNorm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps) {
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float ||
          input.scalar_type() == at::ScalarType::Half ||
          input.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of input tensor of LayerNorm only support Float32, Half and BFloat16 but now it is ",
      input.scalar_type());
  TORCH_CHECK(
      !weight_opt.has_value() || !weight_opt->defined() ||
          weight_opt->scalar_type() == at::ScalarType::Float ||
          weight_opt->scalar_type() == at::ScalarType::Half ||
          weight_opt->scalar_type() == at::ScalarType::BFloat16,
      "Dtype of weight tensor of LayerNorm only support Float32, Half and BFloat16 ",
      "but now it is ",
      weight_opt->scalar_type());
  TORCH_CHECK(
      !bias_opt.has_value() || !bias_opt->defined() ||
          bias_opt->scalar_type() == at::ScalarType::Float ||
          bias_opt->scalar_type() == at::ScalarType::Half ||
          bias_opt->scalar_type() == at::ScalarType::BFloat16,
      "Dtype of bias tensor of LayerNorm only support Float32, Half and BFloat16 ",
      "but now it is ",
      bias_opt->scalar_type());

  const c10::musa::MUSAGuard device_guard(input.device());

  Proxy weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  Proxy bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = at::native::_check_layer_norm_inputs(
      input, normalized_shape, weight, bias);
  auto M = M_N.first;
  Tensor contiguous_input = input.contiguous();
  auto output = at::empty_like(contiguous_input);

  auto mt_input = CreateMUTensor(contiguous_input);
  auto mt_output = CreateMUTensor(output);

  Tensor gamma;
  muTensor mt_gamma;
  if (weight.defined()) {
    TORCH_CHECK(
        output.scalar_type() == weight.scalar_type(),
        "Dtypes of output/weight must be same, but meets ",
        output.scalar_type(),
        " and ",
        weight.scalar_type());
    gamma = weight.contiguous();
    mt_gamma = CreateMUTensor(gamma);
  }

  Tensor beta;
  muTensor mt_beta;
  if (bias.defined()) {
    TORCH_CHECK(
        output.scalar_type() == bias.scalar_type(),
        "Dtypes of output/bias must be same, but meets ",
        output.scalar_type(),
        " and ",
        bias.scalar_type());
    beta = bias.contiguous();
    mt_beta = CreateMUTensor(beta);
  } else if (weight.defined()) {
    // weight != None && bias == None is not supported in muDNN.
    beta = at::zeros_like(weight);
    mt_beta = CreateMUTensor(beta);
  }

  const auto norm_ndim = static_cast<int32_t>(normalized_shape.size());
  const auto ndim = static_cast<int32_t>(input.dim());
  const auto stat_ndim = ndim - norm_ndim;
  std::vector<int32_t> norm_axis(norm_ndim);
  std::iota(norm_axis.begin(), norm_axis.end(), stat_ndim);

  const auto acc_type =
      at::toAccumulateType(input.scalar_type(), /*is_cuda=*/true);
  const auto stat_opt = input.options().dtype(acc_type);
  auto mean = at::empty({M}, stat_opt);
  auto rstd = at::empty({M}, stat_opt);

  const auto input_shape = input.sizes();
  std::vector<int64_t> stat_shape;
  for (const auto idx : c10::irange(stat_ndim)) {
    stat_shape.push_back(input_shape[idx]);
  }
  for (const auto C10_UNUSED idx : c10::irange(norm_ndim)) {
    stat_shape.push_back(1);
  }
  mean = mean.view(stat_shape);
  rstd = rstd.view(stat_shape);
  auto mt_mean = CreateMUTensor(mean);
  auto mt_rstd = CreateMUTensor(rstd);

  if (M > 0) {
    muHandle& h = GetMudnnHandle();
    ::musa::dnn::LayerNorm op;
    CHECK_MUDNN_STATUS(
        op.SetAxis(norm_axis.size(), norm_axis.data()), "SetAxis");
    CHECK_MUDNN_STATUS(op.SetEpsilon(eps), "SetEpsilon");
    MaybeSetVarMode(input, op);
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
  }
  return std::make_tuple(std::move(output), std::move(mean), std::move(rstd));
}

::std::tuple<Tensor, Tensor, Tensor> NativeLayerNormBwd(
    const Tensor& grad_out,
    const Tensor& input,
    at::IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
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

  const c10::musa::MUSAGuard device_guard(input.device());

  Proxy weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  Proxy bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = at::native::_check_layer_norm_inputs(
      input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.contiguous();
  auto gamma = weight.defined() ? weight.contiguous() : weight;
  auto beta = bias.defined() ? bias.contiguous() : bias;

  const auto dx_dtype = X.scalar_type();
  Tensor dX;
  muTensor mt_dX;
  if (grad_input_mask[0]) {
    dX = at::native::empty_like(
        X,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        c10::nullopt /* device */,
        c10::nullopt /* pin_memory */,
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    mt_dX = CreateMUTensor(dX);
  } else {
    SetMUTensorDType(dx_dtype, mt_dX);
  }

  Tensor dgamma, dbeta;
  muTensor mt_dgamma, mt_dbeta;
  if (grad_input_mask[1] || grad_input_mask[2]) {
    const bool has_gamma = gamma.defined();
    const bool has_beta = beta.defined();
    TORCH_INTERNAL_ASSERT(has_gamma || has_beta);

    if (has_gamma) {
      TORCH_CHECK(
          dx_dtype == gamma.scalar_type(),
          "Dtypes of grad_input/weight must be same, but meets ",
          dx_dtype,
          " and ",
          gamma.scalar_type());
    }
    if (has_beta) {
      TORCH_CHECK(
          dx_dtype == beta.scalar_type(),
          "Dtypes of grad_input/bias must be same, but meets ",
          dx_dtype,
          " and ",
          beta.scalar_type());
    }

    const auto& g_ref = has_gamma ? gamma : beta;
    dgamma = M > 0 ? at::native::empty_like(
                         g_ref,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         c10::nullopt /* device */,
                         c10::nullopt /* pin_memory */,
                         LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                   : at::native::zeros_like(
                         g_ref,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         c10::nullopt /* device */,
                         c10::nullopt /* pin_memory */,
                         LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    mt_dgamma = CreateMUTensor(dgamma);

    const auto& b_ref = has_beta ? beta : gamma;
    dbeta = M > 0 ? at::native::empty_like(
                        b_ref,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                  : at::native::zeros_like(
                        b_ref,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        c10::nullopt /* device */,
                        c10::nullopt /* pin_memory */,
                        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    mt_dbeta = CreateMUTensor(dbeta);
  }

  if (M > 0 && N > 0) {
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
    }

    const auto norm_ndim = static_cast<int32_t>(normalized_shape.size());
    const auto ndim = static_cast<int32_t>(input.dim());
    std::vector<int32_t> norm_axis(norm_ndim);
    std::iota(norm_axis.begin(), norm_axis.end(), ndim - norm_ndim);

    muHandle& h = GetMudnnHandle();
    ::musa::dnn::LayerNorm op;
    MaybeSetVarMode(X, op);
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

  if (!grad_input_mask[1]) {
    dgamma = Tensor();
  }
  if (!grad_input_mask[2]) {
    dbeta = Tensor();
  }
  return std::make_tuple(std::move(dX), std::move(dgamma), std::move(dbeta));
}

} // namespace at::musa
