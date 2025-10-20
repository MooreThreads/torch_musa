#include <ATen/AccumulateType.h>
#include <ATen/Config.h>
#include <ATen/TensorOperators.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_native_batch_norm_legit_native.h>
#include <ATen/ops/_native_batch_norm_legit_no_training_native.h>
#include <ATen/ops/batch_norm_update_stats_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/native_batch_norm_backward_native.h>
#include <ATen/ops/native_batch_norm_native.h>
#include <ATen/ops/resize.h>
#endif

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

static inline void resize_out_helper(
    const at::Tensor& dst,
    const at::Tensor& src) {
  at::resize(dst, src.sizes());
}

static void copy_arg(const at::Tensor& dst, const at::Tensor& src) {
  TORCH_CHECK(
      src.dtype() == dst.dtype(),
      "Expected out tensor to have dtype ",
      src.dtype(),
      ", but got ",
      dst.dtype(),
      " instead");
  TORCH_CHECK(
      src.device() == dst.device(),
      "Expected out tensor to have device ",
      src.device(),
      ", but got ",
      dst.device(),
      " instead");
  dst.copy_(src);
}

std::tuple<Tensor&, Tensor&, Tensor&> NativeBatchNormOut(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    bool training,
    double momentum,
    double eps,
    Tensor& output,
    Tensor& save_mean,
    Tensor& save_invstd) {
  const auto inp_dtype = input.scalar_type();
  TORCH_CHECK(
      inp_dtype == at::ScalarType::Float || inp_dtype == at::ScalarType::Half ||
          inp_dtype == at::ScalarType::BFloat16,
      "BatchNorm supports Float/Half/BFloat16 input dtype, now got: ",
      inp_dtype);
  const c10::musa::MUSAGuard device_guard(input.device());
  const auto stat_dtype = save_mean.scalar_type();

  auto vector_to_mutensor = [](const std::optional<Tensor>& vec,
                               const std::optional<ScalarType>& empty_dtype) {
    if (vec.has_value() && vec->defined()) {
      return CreateMUTensor(vec->view({-1}));
    }
    auto ret = CreateEmptyMUTensor();
    if (empty_dtype) {
      SetMUTensorDType(*empty_dtype, ret);
    }
    return ret;
  };

  auto w = vector_to_mutensor(weight_opt, std::nullopt);
  auto b = vector_to_mutensor(bias_opt, std::nullopt);

  auto rm = vector_to_mutensor(running_mean_opt, stat_dtype);
  auto rv = vector_to_mutensor(running_var_opt, stat_dtype);

  auto m = vector_to_mutensor(save_mean, stat_dtype);
  auto v = vector_to_mutensor(save_invstd, stat_dtype);

  const auto input_memory_format = input.suggest_memory_format();
  Tensor contig_input = FormatContiguous(input.alias(), input_memory_format);
  auto in = CreateMUTensor(contig_input);

  auto proxy_out = output.expect_contiguous(input_memory_format);
  auto out = CreateMUTensor(*proxy_out);

  auto& h = GetMudnnHandle();
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
    CHECK_MUDNN_STATUS(bn.RunPure(h, out, in, rm, rv, w, b), "RunPure");
  } else {
    CHECK_MUDNN_STATUS(
        bn.RunComposite(
            h, out, in, rm, rv, m, v, w, b, momentum, InternalMemAlloc),
        "RunComposite");
  }

  if (!(proxy_out->is_same(output))) {
    output.copy_(*proxy_out);
  }
  return std::tuple<Tensor&, Tensor&, Tensor&>(output, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor> NativeBatchNorm(
    const Tensor& self,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps) {
  auto output = at::empty_like(self);
  int64_t n_input = self.size(1);

  auto save_mean_var_dtype =
      (running_mean_opt.has_value() && running_mean_opt->defined())
      ? running_mean_opt->scalar_type()
      : at::toAccumulateType(self.scalar_type(), /*is_cuda=*/true);
  auto options = self.options().dtype(save_mean_var_dtype);
  auto save_mean = at::empty({n_input}, options);
  auto save_invstd = at::empty({n_input}, options);

  NativeBatchNormOut(
      self,
      weight_opt,
      bias_opt,
      running_mean_opt,
      running_var_opt,
      train,
      momentum,
      eps,
      output,
      save_mean,
      save_invstd);

  return std::make_tuple(output, save_mean, save_invstd);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
NativeBatchNormLegitfunctional(
    const at::Tensor& input,
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool training,
    double momentum,
    double eps) {
  auto running_mean_clone = running_mean.clone();
  auto running_var_clone = running_var.clone();
  auto output = NativeBatchNormLegit(
      input,
      weight,
      bias,
      const_cast<Tensor&>(running_mean_clone),
      const_cast<Tensor&>(running_var_clone),
      training,
      momentum,
      eps);
  return ::std::
      tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
          std::get<0>(output),
          std::get<1>(output),
          std::get<2>(output),
          running_mean_clone,
          running_var_clone);
}

std::tuple<Tensor, Tensor, Tensor> NativeBatchNormLegit(
    const Tensor& self,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    Tensor& running_mean,
    Tensor& running_var,
    bool train,
    double momentum,
    double epsilon) {
  return NativeBatchNorm(
      self,
      weight_opt,
      bias_opt,
      running_mean,
      running_var,
      train,
      momentum,
      epsilon);
}

std::tuple<Tensor, Tensor, Tensor> NativeBatchNormLegitNoTraining(
    const Tensor& self,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum,
    double eps) {
  return NativeBatchNormLegit(
      self,
      weight_opt,
      bias_opt,
      const_cast<Tensor&>(running_mean),
      const_cast<Tensor&>(running_var),
      /*train=*/false,
      momentum,
      eps);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>
NativeBatchNormLegitNoTrainingOut(
    const at::Tensor& input,
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    double momentum,
    double eps,
    at::Tensor& out0,
    at::Tensor& out1,
    at::Tensor& out2) {
  auto tmp_output = NativeBatchNormLegitNoTraining(
      input, weight, bias, running_mean, running_var, momentum, eps);
  resize_out_helper(out0, std::get<0>(tmp_output));
  copy_arg(out0, std::get<0>(tmp_output));
  resize_out_helper(out1, std::get<1>(tmp_output));
  copy_arg(out1, std::get<1>(tmp_output));
  resize_out_helper(out2, std::get<2>(tmp_output));
  copy_arg(out2, std::get<2>(tmp_output));
  return ::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>(out0, out1, out2);
}

std::tuple<Tensor&, Tensor&, Tensor&> NativeBatchNormLegitOut(
    const Tensor& self,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    Tensor& running_mean,
    Tensor& running_var,
    bool train,
    double momentum,
    double epsilon,
    Tensor& output,
    Tensor& save_mean,
    Tensor& save_invstd) {
  return NativeBatchNormOut(
      self,
      weight_opt,
      bias_opt,
      running_mean,
      running_var,
      train,
      momentum,
      epsilon,
      output,
      save_mean,
      save_invstd);
}

std::tuple<Tensor, Tensor, Tensor> NativeBatchNormLegitNoStats(
    const Tensor& self,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    bool train,
    double momentum,
    double epsilon) {
  return NativeBatchNorm(
      self, weight_opt, bias_opt, Tensor(), Tensor(), train, momentum, epsilon);
}

std::tuple<Tensor&, Tensor&, Tensor&> NativeBatchNormLegitNoStatsOut(
    const Tensor& self,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    bool train,
    double momentum,
    double epsilon,
    Tensor& output,
    Tensor& save_mean,
    Tensor& save_invstd) {
  return NativeBatchNormOut(
      self,
      weight_opt,
      bias_opt,
      Tensor(),
      Tensor(),
      train,
      momentum,
      epsilon,
      output,
      save_mean,
      save_invstd);
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

  if (output_mask[1] || output_mask[2]) {
    grad_weight = at::empty_like(weight);
    grad_bias = at::empty_like(weight);
  }

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

} // namespace musa
} // namespace at
