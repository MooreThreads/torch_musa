#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/layer_norm.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace native {
namespace musa {

void check_dims_match_num_input_features(
    const char* arg_name,
    int64_t expected,
    int64_t actual) {
  TORCH_CHECK(
      actual == expected,
      arg_name,
      " should contain ",
      expected,
      " elements not ",
      actual);
}

std::tuple<Tensor, Tensor, Tensor> NativeBatchNorm(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    bool training,
    double momentum,
    double eps) {
  const Tensor& weight =
      c10::value_or_else(weight_opt, [] { return Tensor(); });
  const Tensor& bias = c10::value_or_else(bias_opt, [] { return Tensor(); });

  // Copy from Normalization.cpp : _batch_norm_impl_index
  auto num_features = input.size(1);
  if (input.numel() == 0) {
    auto out = input.clone();
    auto options = input.options();
    auto save_mean = at::empty({num_features}, options);
    auto save_invstd = at::empty({num_features}, options);
    if (weight.defined()) {
      out = out * weight[0];
    }
    if (bias.defined()) {
      out = out + bias[0];
    }
    return std::tuple<Tensor&, Tensor&, Tensor&>{out, save_mean, save_invstd};
  }

  const Tensor& running_mean =
      c10::value_or_else(running_mean_opt, [] { return Tensor(); });
  const Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });

  // Copy from Normalization.cpp : _batch_norm_impl_index
  if (running_mean.defined()) {
    check_dims_match_num_input_features(
        "running_mean", num_features, running_mean.numel());
  } else if (!training) {
    AT_ERROR("running_mean must be defined in evaluation mode");
  }
  if (running_var.defined()) {
    check_dims_match_num_input_features(
        "running_var", num_features, running_var.numel());
  } else if (!training) {
    AT_ERROR("running_var must be defined in evaluation mode");
  }
  if (weight.defined()) {
    check_dims_match_num_input_features("weight", num_features, weight.numel());
  }
  if (bias.defined()) {
    check_dims_match_num_input_features("bias", num_features, bias.numel());
  }

  auto options = input.options().dtype(input.scalar_type());
  auto output = at::empty_like(input);
  auto out = CreateMUTensor(output);
  auto in = CreateMUTensor(input);
  muTensor s;
  muTensor b;
  if (weight.defined()) {
    s = CreateMUTensor(weight);
    b = CreateMUTensor(bias);
  }
  muTensor am;
  muTensor av;
  // In case of track_running_stats is False
  if (running_mean.defined()) {
    am = CreateMUTensor(running_mean);
    av = CreateMUTensor(running_var);
  }

  auto contiguous_input = input;
  ConfigFormat(contiguous_input, in, true);
  ConfigFormat(output, out, true);

  muHandle h;
  ::musa::dnn::BatchNorm bn;
  CHECK_MUDNN_STATUS(bn.SetEpsilon(eps), "SetEpsilon");

  auto save_mean = at::empty({0}, options);
  auto save_invstd = at::empty({0}, options);

  if (!training) {
    CHECK_MUDNN_STATUS(bn.RunPure(h, out, in, am, av, s, b), "RunPure");
  } else {
    save_mean = at::empty({num_features}, options);
    save_invstd = at::empty({num_features}, options);
    auto m = CreateMUTensor(save_mean);
    auto v = CreateMUTensor(save_invstd);

    CHECK_MUDNN_STATUS(
        bn.RunComposite(h, out, in, am, av, m, v, s, b, momentum),
        "RunComposite");
  }
  return std::tuple<Tensor&, Tensor&, Tensor&>{output, save_mean, save_invstd};
}

std::tuple<Tensor, Tensor, Tensor> NativeBatchNormBwd(
    const Tensor& grad_out,
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& running_mean_opt /* optional */,
    const c10::optional<Tensor>& running_var_opt /* optional */,
    const c10::optional<Tensor>& save_mean_opt /* optional */,
    const c10::optional<Tensor>& save_invstd_opt /* optional */,
    bool train,
    double eps,
    std::array<bool, 3> output_mask) {
  // Copy from Normalization.cpp : _batch_norm_impl_index_backward
  TORCH_CHECK(
      grad_out.device().type() == kMUSA,
      "Device of grad_output tensor of BatchNormBackward must be MUSA, ",
      "but now is ",
      grad_out.device());
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of BatchNormBackward must be MUSA, but now is ",
      input.device());
  TORCH_CHECK(
      grad_out.scalar_type() == at::ScalarType::Float,
      "Dtype of grad_out tensor of BatchNormBackward only support Float32, ",
      "but now it is ",
      grad_out.scalar_type());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of BatchNormBackward only support Float32, ",
      "but now it is ",
      input.scalar_type());
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& running_mean =
      c10::value_or_else(running_mean_opt, [] { return Tensor(); });
  const Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });
  const Tensor& save_mean =
      c10::value_or_else(save_mean_opt, [] { return Tensor(); });
  const Tensor& save_invstd =
      c10::value_or_else(save_invstd_opt, [] { return Tensor(); });

  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  if (input.numel() == 0) {
    std::vector<int64_t> dims(input.dim() - 1);
    dims[0] = 0;
    std::iota(dims.begin() + 1, dims.end(), 2);

    // don't return empty tensor because it will break gradient chain
    if (output_mask[2]) {
      grad_bias = grad_out.sum(dims);
    }
    if (output_mask[1]) {
      grad_weight = (grad_out * input).sum(dims);
    }
    if (output_mask[0] && weight.defined()) {
      grad_input = grad_out * weight[0];
    }
    return std::make_tuple(grad_input, grad_weight, grad_bias);
  }

  auto num_features = input.size(1);
  auto options = input.options().dtype(input.scalar_type());
  grad_input = at::empty_like(input, options);
  auto grad_mean = at::empty_like(save_mean, options);
  auto grad_var = at::empty_like(save_invstd, options);
  grad_weight = at::empty({num_features}, options);
  grad_bias = at::empty({num_features}, options);

  auto dx = CreateMUTensor(grad_input);
  auto dm = CreateMUTensor(grad_mean);
  auto dv = CreateMUTensor(grad_var);
  auto dg = CreateMUTensor(grad_weight);
  auto db = CreateMUTensor(grad_bias);
  auto x = CreateMUTensor(input);
  auto dy = CreateMUTensor(grad_out);
  auto m = CreateMUTensor(save_mean);
  auto v = CreateMUTensor(save_invstd);
  auto g = CreateMUTensor(weight);

  auto contiguous_input = input;
  auto grad_out_ = grad_out;
  ConfigFormat(contiguous_input, x, true);
  ConfigFormat(grad_out_, dy, true);
  ConfigFormat(grad_input, dx, true);
  ConfigFormat(grad_weight, dg, true);
  ConfigFormat(grad_bias, db, true);
  ConfigFormat(grad_mean, dm, true);
  ConfigFormat(grad_var, dv, true);

  muHandle h;
  ::musa::dnn::BatchNorm bn;
  CHECK_MUDNN_STATUS(bn.SetEpsilon(eps), "BN SetEpsilon");
  CHECK_MUDNN_STATUS(bn.SetTraining(train), "BN SetTraining");

  CHECK_MUDNN_STATUS(
      bn.RunBwd(h, dx, dm, dv, dg, db, x, dy, m, v, g), "BN RunBwd");
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor> NativeLayerNorm(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */,
    double eps) {
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of NativeLayerNorm must be MUSA, but now is ",
      input.device());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of LayerNorm only support Float32, but now it is ",
      input.scalar_type());
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  Tensor input_contiguous = Contiguous(input);
  auto output = at::empty_like(input_contiguous);

  muHandle h;
  ::musa::dnn::LayerNorm op;
  auto mt_input = CreateMUTensor(input_contiguous);
  auto mt_output = CreateMUTensor(output);
  muTensor mt_gamma;
  muTensor mt_beta;
  if (weight.defined()) {
    auto gamma = Contiguous(weight);
    mt_gamma = CreateMUTensor(gamma);
    TORCH_CHECK(
        weight.device().type() == kMUSA,
        "Device of weight tensor of NativeLayerNorm must be MUSA, but now is ",
        weight.device());
    TORCH_CHECK(
        weight.scalar_type() == at::ScalarType::Float,
        "Dtype of weight tensor of LayerNorm only support Float32, ",
        "but now it is ",
        weight.scalar_type());
  }
  if (bias.defined()) {
    auto beta = Contiguous(bias);
    mt_beta = CreateMUTensor(beta);
    TORCH_CHECK(
        bias.device().type() == kMUSA,
        "Device of bias tensor of NativeLayerNorm must be MUSA, but now is ",
        bias.device());
    TORCH_CHECK(
        bias.scalar_type() == at::ScalarType::Float,
        "Dtype of bias tensor of LayerNorm only support Float32, ",
        "but now it is ",
        bias.scalar_type());
  }
  ConfigFormat(input_contiguous, mt_input, true);
  ConfigFormat(output, mt_output, true);

  std::vector<int32_t> norm_axis;
  const int32_t diff = input.dim() - normalized_shape.size();
  for (size_t i = 0; i != normalized_shape.size(); ++i) {
    if (input.size(diff + i) == normalized_shape[i]) {
      norm_axis.push_back(diff + i);
    }
  }
  CHECK_MUDNN_STATUS(op.SetAxis(norm_axis.size(), norm_axis.data()), "SetAxis");
  CHECK_MUDNN_STATUS(op.SetEpsilon(eps), "SetEpsilon");

  const auto input_shape = input.sizes();
  const int32_t axis = input.dim() - normalized_shape.size();
  auto mean = at::empty({M}, input.options());
  auto rstd = at::empty({M}, input.options());
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
      grad_out.device().type() == kMUSA,
      "Device of grad_output tensor of LayerNormBackward must be MUSA, ",
      "but now is ",
      grad_out.device());
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of LayerNormBackward must be MUSA, but now is ",
      input.device());
  TORCH_CHECK(
      mean.device().type() == kMUSA,
      "Device of mean tensor of LayerNormBackward must be MUSA, but now is ",
      mean.device());
  TORCH_CHECK(
      rstd.device().type() == kMUSA,
      "Device of rstd tensor of LayerNormBackward must be MUSA, but now is ",
      rstd.device());
  TORCH_CHECK(
      grad_out.scalar_type() == at::ScalarType::Float,
      "Dtype of grad_out tensor of LayerNormBackward only support Float32, ",
      "but now it is ",
      grad_out.scalar_type());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of LayerNormBackward only support Float32, ",
      "but now it is ",
      input.scalar_type());
  TORCH_CHECK(
      mean.scalar_type() == at::ScalarType::Float,
      "Dtype of mean tensor of LayerNormBackward only support Float32, ",
      "but now it is ",
      mean.scalar_type());
  TORCH_CHECK(
      rstd.scalar_type() == at::ScalarType::Float,
      "Dtype of rstd tensor of LayerNormBackward only support Float32, ",
      "but now it is ",
      rstd.scalar_type());
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto X = Contiguous(input);
  auto gamma = weight.defined() ? Contiguous(weight) : weight;
  auto beta = bias.defined() ? Contiguous(bias) : bias;

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
    auto contiguous_grad_out = Contiguous(grad_out);
    auto contiguous_mean = Contiguous(mean);
    auto contiguous_rstd = Contiguous(rstd);
    auto mt_grad_out = CreateMUTensor(contiguous_grad_out);
    auto mt_X = CreateMUTensor(X);
    auto mt_mean = CreateMUTensor(contiguous_mean);
    auto mt_rstd = CreateMUTensor(contiguous_rstd);
    muTensor mt_weight;
    if (weight.defined()) {
      mt_weight = CreateMUTensor(gamma);
      TORCH_CHECK(
          weight.device().type() == kMUSA,
          "Device of weight tensor of LayerNormBackward must be MUSA, ",
          "but now is ",
          weight.device());
      TORCH_CHECK(
          weight.scalar_type() == at::ScalarType::Float,
          "Dtype of weight tensor of LayerNormBackward only support Float32, ",
          "but now it is ",
          weight.scalar_type());
    }
    muHandle h;
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

void check_group_norm_inputs(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t C,
    int64_t num_groups) {
  TORCH_CHECK(
      num_groups > 0,
      "Expected num groups to be greater than 0, got ",
      num_groups);
  TORCH_CHECK(
      C % num_groups == 0,
      "Expected number of channels in input to be divisible by ",
      "num_groups, but got input of shape ",
      input.sizes(),
      " and "
      "num_groups=",
      num_groups);
  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && weight.numel() == C),
      "Expected weight to be a vector of size equal to the number of ",
      "channels in input, but got weight of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());
  TORCH_CHECK(
      !bias.defined() || (bias.dim() == 1 && bias.numel() == C),
      "Expected bias to be a vector of size equal to the number of ",
      "channels in input, but got bias of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());
}

std::tuple<Tensor, Tensor, Tensor> NativeGroupNorm(
    const Tensor& X,
    const c10::optional<Tensor>& gamma_opt /* optional */,
    const c10::optional<Tensor>& beta_opt /* optional */,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps) {
  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor& contiguous_gamma = Contiguous(*gamma_maybe_owned);
  const Tensor& contiguous_beta =
      Contiguous(c10::value_or_else(beta_opt, [] { return Tensor(); }));

  check_group_norm_inputs(X, contiguous_gamma, contiguous_beta, C, group);

  Tensor contiguous_X = Contiguous(X);
  Tensor contiguous_Y = at::native::empty_like(contiguous_X);
  Tensor contiguous_mean = at::empty({N, group}, contiguous_X.options());
  Tensor contiguous_rstd = at::empty({N, group}, contiguous_X.options());

  auto in = CreateMUTensor(contiguous_X);
  auto out = CreateMUTensor(contiguous_Y);
  auto mean = CreateMUTensor(contiguous_mean);
  auto rstd = CreateMUTensor(contiguous_rstd);

  muTensor gamma = contiguous_gamma.defined() ? CreateMUTensor(contiguous_gamma)
                                              : muTensor();
  muTensor beta =
      contiguous_beta.defined() ? CreateMUTensor(contiguous_beta) : muTensor();

  muHandle h;
  ::musa::dnn::GroupNorm op;
  CHECK_MUDNN_STATUS(op.SetEpsilon(eps), "SetEpsilon");
  CHECK_MUDNN_STATUS(op.SetAxis(1), "SetAxis");
  CHECK_MUDNN_STATUS(op.SetGroup(static_cast<int>(group)), "SetGroup");
  CHECK_MUDNN_STATUS(op.Run(h, out, mean, rstd, in, gamma, beta), "RunOp");

  return std::make_tuple(contiguous_Y, contiguous_mean, contiguous_rstd);
}

std::tuple<Tensor, Tensor, Tensor> NativeGroupNormBwd(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const c10::optional<Tensor>& gamma_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    std::array<bool, 3> grad_input_mask) {
  auto gamma_cpu = gamma_opt.has_value() ? (*gamma_opt).to("cpu") : Tensor();
  auto result = at::native_group_norm_backward(
      dY.to("cpu"),
      X.to("cpu"),
      mean.to("cpu"),
      rstd.to("cpu"),
      gamma_cpu,
      N,
      C,
      HxW,
      group,
      grad_input_mask);
  return {
      std::get<0>(result).to("musa"),
      std::get<1>(result).to("musa"),
      std::get<2>(result).to("musa")};
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("native_batch_norm", &NativeBatchNorm);
  m.impl("native_batch_norm_backward", &NativeBatchNormBwd);
  m.impl("native_layer_norm", &NativeLayerNorm);
  m.impl("native_layer_norm_backward", &NativeLayerNormBwd);
  m.impl("native_group_norm", &NativeGroupNorm);
  m.impl("native_group_norm_backward", &NativeGroupNormBwd);
}

} // namespace musa
} // namespace native
} // namespace at
