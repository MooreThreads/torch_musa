#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/layer_norm.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

#include <mudnn.h>

namespace at {
namespace musa {

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
  (void)HxW;
  TORCH_CHECK(
      X.scalar_type() == at::ScalarType::Float ||
          X.scalar_type() == at::ScalarType::Half,
      "group norm supports Float and Half tensor type, now got ",
      X.scalar_type());

  c10::musa::MUSAGuard device_guard(X.device());
  c10::MaybeOwned<Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const Tensor& gamma_ref = *gamma_maybe_owned;
  c10::MaybeOwned<Tensor> beta_maybe_owned =
      at::borrow_from_optional_tensor(beta_opt);
  const Tensor& beta_ref = *beta_maybe_owned;

  check_group_norm_inputs(X, gamma_ref, beta_ref, C, group);

  Tensor contiguous_X = X.contiguous();
  Tensor contiguous_Y = at::native::empty_like(contiguous_X);
  Tensor contiguous_mean = at::empty({N, group}, contiguous_X.options());
  Tensor contiguous_rstd = at::empty({N, group}, contiguous_X.options());
  Tensor contiguous_gamma;
  Tensor contiguous_beta;

  muTensor in = CreateMUTensor(contiguous_X);
  muTensor out = CreateMUTensor(contiguous_Y);
  muTensor mean = CreateMUTensor(contiguous_mean);
  muTensor rstd = CreateMUTensor(contiguous_rstd);
  muTensor gamma = muTensor();
  muTensor beta = muTensor();
  if (gamma_ref.defined()) {
    contiguous_gamma = gamma_ref.contiguous();
    gamma = CreateMUTensor(contiguous_gamma);
  }
  if (beta_ref.defined()) {
    contiguous_beta = beta_ref.contiguous();
    beta = CreateMUTensor(contiguous_beta);
  }

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::GroupNorm op;
  CHECK_MUDNN_STATUS(op.SetEpsilon(eps), "SetEpsilon");
  CHECK_MUDNN_STATUS(op.SetAxis(1), "SetAxis");
  CHECK_MUDNN_STATUS(op.SetGroup(static_cast<int>(group)), "SetGroup");
  CHECK_MUDNN_STATUS(op.Run(h, out, mean, rstd, in, gamma, beta), "RunOp");

  return std::make_tuple(contiguous_Y, contiguous_mean, contiguous_rstd);
}

std::tuple<Tensor, Tensor, Tensor> NativeGroupNormBwd(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const c10::optional<Tensor>& weight,
    c10::SymInt N,
    c10::SymInt C,
    c10::SymInt HxW,
    int64_t group,
    ::std::array<bool, 3> output_mask) {
  c10::musa::MUSAGuard device_guard(grad_out.device());
  return at::native::native_group_norm_backward(
      grad_out,
      input,
      mean,
      rstd,
      weight,
      N.expect_int(),
      C.expect_int(),
      HxW.expect_int(),
      group,
      output_mask);
}

ADVANCED_REGISTER(aten, PrivateUse1, "native_group_norm", NativeGroupNorm)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "native_group_norm_backward",
    NativeGroupNormBwd)

} // namespace musa
} // namespace at
