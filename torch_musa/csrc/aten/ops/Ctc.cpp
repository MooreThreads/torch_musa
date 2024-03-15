#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Activation.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <limits>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

// MUSA CTC forward
::std::tuple<Tensor, Tensor> CtcLoss(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t blank,
    bool zero_infinity) {
  c10::musa::MUSAGuard device_guard(log_probs.device());
  return at::native::ctc_loss_gpu(
      log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
}

// MUSA CTC backward
Tensor CtcLossBackward(
    const Tensor& grad,
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    const Tensor& neg_log_likelihood,
    const Tensor& log_alpha,
    int64_t blank,
    bool zero_infinity) {
  c10::musa::MUSAGuard device_guard(grad.device());
  return at::native::ctc_loss_backward_gpu(
      grad,
      log_probs,
      targets,
      input_lengths,
      target_lengths,
      neg_log_likelihood,
      log_alpha,
      blank,
      zero_infinity);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_ctc_loss", &CtcLoss);
  m.impl("_ctc_loss_backward", &CtcLossBackward);
}

} // namespace musa
} // namespace at
