#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/layer_norm.h>
#include <torch/library.h>
#pragma GCC diagnostic pop

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

::std::tuple<Tensor, Tensor, Tensor> BatchNorm(
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

std::tuple<Tensor, Tensor, Tensor> BatchNormBwd(
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
      "Device of grad_output tensor of BatchNormBackward must be MTGPU, ",
      "but now is ",
      grad_out.device());
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of BatchNormBackward must be MTGPU, but now is ",
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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("native_batch_norm", &BatchNorm);
  m.impl("native_batch_norm_backward", &BatchNormBwd);
}

} // namespace musa
} // namespace native
} // namespace at
