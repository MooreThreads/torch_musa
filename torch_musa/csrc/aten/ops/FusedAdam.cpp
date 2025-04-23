#include <ATen/TypeDefault.h>
#include <ATen/native/ForeachUtils.h>
#include <c10/util/Exception.h>

namespace at::musa {
void FusedAdamKernel(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  TORCH_CHECK(
      false,
      "torch.optim.Adam(fused=True) is not supported, try torch_musa.optim.FusedAdam instead");
}

// The following overload simply has a Tensor lr
void FusedAdamKernel(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  TORCH_CHECK(
      false,
      "torch.optim.Adam(fused=True) is not supported, try torch_musa.optim.FusedAdam instead");
}

} // namespace at::musa
