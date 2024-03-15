#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/layer_norm.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

#include <mudnn.h>

namespace at {
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
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float ||
          input.scalar_type() == at::ScalarType::Half ||
          input.scalar_type() == at::ScalarType::BFloat16,
      "batch_norm supports Float or Half/BFloat16 tensor dtype, now got: ",
      input.scalar_type());

  const c10::musa::MUSAGuard device_guard(input.device());
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

  const auto input_memory_format = input.suggest_memory_format();
  Tensor contiguous_input = FormatContiguous(input, input_memory_format);
  auto options = input.options().dtype(input.scalar_type());

  Tensor output = at::empty_like(contiguous_input);
  Tensor contiguous_weight;
  Tensor contiguous_bias;
  Tensor contiguous_running_mean;
  Tensor contiguous_running_var;
  Tensor save_mean = at::empty({0}, options);
  Tensor save_invstd = at::empty({0}, options);

  // Copy from Normalization.cpp : _batch_norm_impl_index
  if (running_mean.defined()) {
    check_dims_match_num_input_features(
        "running_mean", num_features, running_mean.numel());
    contiguous_running_mean = running_mean.contiguous();
  } else if (!training) {
    AT_ERROR("running_mean must be defined in evaluation mode");
  }
  if (running_var.defined()) {
    check_dims_match_num_input_features(
        "running_var", num_features, running_var.numel());
    contiguous_running_var = running_var.contiguous();
  } else if (!training) {
    AT_ERROR("running_var must be defined in evaluation mode");
  }
  if (weight.defined()) {
    check_dims_match_num_input_features("weight", num_features, weight.numel());
  }
  if (bias.defined()) {
    check_dims_match_num_input_features("bias", num_features, bias.numel());
  }

  // computational muTensors
  muTensor in = CreateMUTensor(contiguous_input);
  muTensor out = CreateMUTensor(output);
  muTensor scale;
  muTensor bias_;
  muTensor mean;
  muTensor variance;

  if (weight.defined()) {
    scale = CreateMUTensor(ContiguousRef(weight, contiguous_weight));
    bias_ = CreateMUTensor(ContiguousRef(bias, contiguous_bias));
  }
  // In case of track_running_stats is False
  if (running_mean.defined()) {
    mean = CreateMUTensor(contiguous_running_mean);
    variance = CreateMUTensor(contiguous_running_var);
  }

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::BatchNorm bn;
  CHECK_MUDNN_STATUS(bn.SetEpsilon(eps), "SetEpsilon");
  CHECK_MUDNN_STATUS(bn.SetTraining(training), "SetTraining");
  // muDNN supports PER_CHANNEL and PER_ACTIVATION modes, while PER_ACTIVATION
  // has higher performance and lower accuracy, we hard code PER_CHANNEL here to
  // keep the accuracy (BN can be folded into conv easily which could speed up
  // inference)
  CHECK_MUDNN_STATUS(
      bn.SetMode(::musa::dnn::BatchNorm::Mode::PER_CHANNEL), "SetTraining");

  if (!training) {
    CHECK_MUDNN_STATUS(
        bn.RunPure(h, out, in, mean, variance, scale, bias_), "RunPure");
  } else {
    // mean/variance/cur_mean/cur_var dtype are consistent in mudnn.
    if (running_mean.defined()) {
      options = contiguous_running_mean.options();
    }
    save_mean =
        at::empty({num_features}, options, at::MemoryFormat::Contiguous);
    save_invstd =
        at::empty({num_features}, options, at::MemoryFormat::Contiguous);
    muTensor cur_mean = CreateMUTensor(save_mean);
    muTensor cur_var = CreateMUTensor(save_invstd);

    CHECK_MUDNN_STATUS(
        bn.RunComposite(
            h,
            out,
            in,
            mean,
            variance,
            cur_mean,
            cur_var,
            scale,
            bias_,
            momentum,
            InternalMemAlloc),
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
      grad_out.scalar_type() == at::ScalarType::Float ||
          grad_out.scalar_type() == at::ScalarType::Half ||
          grad_out.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of grad_out tensor of BatchNormBackward only support Float32/Half/Bfloat16, ",
      "but now it is ",
      grad_out.scalar_type());
  TORCH_CHECK(
      grad_out.scalar_type() == input.scalar_type(),
      "Dtype of grad_out and input tensor of BatchNormBackward must be same, ",
      "but now it is ",
      grad_out.scalar_type(),
      input.scalar_type());

  const c10::musa::MUSAGuard device_guard(input.device());
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> save_mean =
      at::borrow_from_optional_tensor(save_mean_opt);
  c10::MaybeOwned<Tensor> save_invstd =
      at::borrow_from_optional_tensor(save_invstd_opt);
  c10::MaybeOwned<Tensor> running_mean =
      at::borrow_from_optional_tensor(running_mean_opt);
  c10::MaybeOwned<Tensor> running_var =
      at::borrow_from_optional_tensor(running_var_opt);
  const bool needs_reduction = train || output_mask[1] || output_mask[2];
  Tensor mean;
  TORCH_INTERNAL_ASSERT(
      save_mean->defined(), "save_mean should always be defined\n");
  if (save_mean->numel() != 0) {
    mean = *save_mean;
  } else if (needs_reduction) {
    TORCH_CHECK(!train && running_mean->defined());
    mean = *running_mean;
  }
  Tensor invstd;
  TORCH_INTERNAL_ASSERT(
      save_invstd->defined(), "save_invstd should always be defined\n");
  if (save_invstd->numel() != 0) {
    invstd = *save_invstd;
  } else {
    TORCH_CHECK(!train && running_var->defined());
    auto n_channels = input.sizes()[1];
    invstd = *running_var;
  }

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

  grad_input = at::empty_like(input);
  auto grad_mean = at::empty_like(mean);
  auto grad_var = at::empty_like(invstd);
  auto weight_options = weight.options().dtype(weight.scalar_type());
  grad_weight =
      at::empty({num_features}, weight_options, at::MemoryFormat::Contiguous);
  grad_bias =
      at::empty({num_features}, weight_options, at::MemoryFormat::Contiguous);

  auto dx = CreateMUTensor(grad_input);
  auto dm = CreateMUTensor(grad_mean);
  auto dv = CreateMUTensor(grad_var);
  auto dg = CreateMUTensor(grad_weight);
  auto db = CreateMUTensor(grad_bias);

  const auto input_memory_format = input.suggest_memory_format();
  auto contiguous_input = FormatContiguous(input, input_memory_format);
  auto x = CreateMUTensor(contiguous_input);

  auto contiguous_grad_out = FormatContiguous(grad_out, input_memory_format);
  auto dy = CreateMUTensor(contiguous_grad_out);

  auto m = CreateMUTensor(mean);
  auto v = CreateMUTensor(invstd);
  auto g = CreateMUTensor(weight);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::BatchNorm bn;
  CHECK_MUDNN_STATUS(bn.SetEpsilon(eps), "BN SetEpsilon");
  CHECK_MUDNN_STATUS(bn.SetTraining(train), "BN SetTraining");

  CHECK_MUDNN_STATUS(
      bn.RunBwd(h, dx, dm, dv, dg, db, x, dy, m, v, g, InternalMemAlloc),
      "BN RunBwd");
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

ADVANCED_REGISTER(aten, PrivateUse1, "native_batch_norm", NativeBatchNorm)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "native_batch_norm_backward",
    NativeBatchNormBwd)

} // namespace musa
} // namespace at
