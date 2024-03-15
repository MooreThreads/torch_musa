#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Activation.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <limits>
#include <numeric>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

#include <mudnn.h>

namespace at {
namespace musa {

::std::tuple<Tensor, Tensor> CtcLoss(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t blank,
    bool zero_infinity) {
  TORCH_CHECK(
      log_probs.device().type() == kMUSA,
      "Device of log_probs tensor of CtcLoss must be MUSA, "
      "but now is ",
      log_probs.device());
  TORCH_CHECK(
      targets.device().type() == kMUSA,
      "Device of targets tensor of CtcLoss must be MUSA, "
      "but now is ",
      targets.device());

  c10::musa::MUSAGuard device_guard(log_probs.device());

  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  TORCH_CHECK(
      (0 <= blank) && (blank < num_labels), "blank must be in label range");
  TORCH_CHECK(
      input_lengths.size() == batch_size,
      "input_lengths must be of size batch_size");
  TORCH_CHECK(
      target_lengths.size() == batch_size,
      "target_lengths must be of size batch_size");

  // two cases, concatenated targets and batch x max_target_length
  auto* tgt_lengths_data = target_lengths.data();
  int64_t max_target_length =
      *std::max_element(tgt_lengths_data, tgt_lengths_data + batch_size);

  auto target_lengths_t =
      at::tensor(target_lengths, targets.options().dtype(torch::kLong));
  auto input_lengths_t =
      at::tensor(input_lengths, targets.options().dtype(torch::kLong));

  Tensor log_alpha = at::empty(
      {batch_size, log_probs.size(0), 2 * max_target_length + 1},
      log_probs.options(),
      at::MemoryFormat::Contiguous);
  Tensor neg_log_likelihood = at::empty(
      {batch_size}, log_probs.options(), at::MemoryFormat::Contiguous);

  const Tensor& contiguous_log_probs = log_probs.contiguous();
  const Tensor& contiguous_targets = targets.contiguous();

  Tensor contiguous_input_lengths = input_lengths_t.contiguous();
  Tensor contiguous_target_lengths = target_lengths_t.contiguous();

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::CTCLoss op;

  auto mt_log_probs = CreateMUTensor(contiguous_log_probs);
  auto mt_targets = CreateMUTensor(contiguous_targets);

  auto mt_input_lengths = CreateMUTensor(contiguous_input_lengths);
  auto mt_target_lengths = CreateMUTensor(contiguous_target_lengths);

  auto mt_neg_log_likelihood = CreateMUTensor(neg_log_likelihood);
  auto mt_log_alpha = CreateMUTensor(log_alpha);

  CHECK_MUDNN_STATUS(op.SetBlank(blank), "SetBlank");
  CHECK_MUDNN_STATUS(op.SetZeroInfinity(zero_infinity), "SetZeroInfinity");
  CHECK_MUDNN_STATUS(
      op.SetMaxTargetLength(max_target_length), "SetMaxTargetLenght");

  CHECK_MUDNN_STATUS(
      op.Run(
          h,
          mt_neg_log_likelihood,
          mt_log_alpha,
          mt_log_probs,
          mt_targets,
          mt_input_lengths,
          mt_target_lengths),
      "Run");
  return std::make_tuple(neg_log_likelihood, log_alpha);
}

template <typename scalar_t>
Tensor CtcLossBackwardImpl(
    const Tensor& grad,
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    const Tensor& neg_log_likelihood,
    const Tensor& log_alpha,
    int64_t blank,
    bool zero_infinity) {
  TORCH_CHECK(
      grad.device().type() == kMUSA,
      "Device of grad tensor of CtcLoss Backward must be MUSA, "
      "but now is ",
      grad.device());
  TORCH_CHECK(
      log_probs.device().type() == kMUSA,
      "Device of log_probs tensor of CtcLoss Backward must be MUSA, "
      "but now is ",
      log_probs.device());
  TORCH_CHECK(
      targets.device().type() == kMUSA,
      "Device of targets tensor of CtcLoss Backward must be MUSA, "
      "but now is ",
      targets.device());
  TORCH_CHECK(
      neg_log_likelihood.device().type() == kMUSA,
      "Device of neg_log_likelihood tensor of CtcLoss Backward must be MUSA, "
      "but now is ",
      neg_log_likelihood.device());
  TORCH_CHECK(
      log_alpha.device().type() == kMUSA,
      "Device of log_alpha tensor of CtcLoss Backward must be MUSA, "
      "but now is ",
      log_alpha.device());

  c10::musa::MUSAGuard device_guard(grad.device());

  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();

  auto target_lengths_t =
      at::tensor(target_lengths, targets.options().dtype(torch::kLong));
  auto input_lengths_t =
      at::tensor(input_lengths, targets.options().dtype(torch::kLong));

  Tensor result =
      at::full_like(log_probs, neginf, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  auto* tgt_lengths_data = target_lengths.data();
  int64_t max_target_length =
      *std::max_element(tgt_lengths_data, tgt_lengths_data + log_probs.size(1));

  const Tensor& contiguous_grad = grad.contiguous();
  const Tensor& contiguous_log_probs = log_probs.contiguous();
  const Tensor& contiguous_targets = targets.contiguous();
  const Tensor& contiguous_loss = neg_log_likelihood.contiguous();
  const Tensor& contiguous_alpha = log_alpha.contiguous();
  Tensor contiguous_target_lengths_t = target_lengths_t.contiguous();
  Tensor contiguous_input_lengths_t = input_lengths_t.contiguous();

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::CTCLoss op;

  auto mt_grad = CreateMUTensor(contiguous_grad);
  auto mt_log_probs = CreateMUTensor(contiguous_log_probs);
  auto mt_targets = CreateMUTensor(contiguous_targets);
  auto mt_loss = CreateMUTensor(contiguous_loss);
  auto mt_alpha = CreateMUTensor(contiguous_alpha);
  auto mt_input_lengths = CreateMUTensor(contiguous_input_lengths_t);
  auto mt_target_lengths = CreateMUTensor(contiguous_target_lengths_t);

  auto mt_grad_input = CreateMUTensor(result);

  CHECK_MUDNN_STATUS(op.SetBlank(blank), "SetBlank");
  CHECK_MUDNN_STATUS(op.SetZeroInfinity(zero_infinity), "SetZeroInfinity");
  CHECK_MUDNN_STATUS(
      op.SetMaxTargetLength(max_target_length), "SetMaxTargetLength");

  CHECK_MUDNN_STATUS(
      op.RunBwd(
          h,
          mt_grad_input,
          mt_grad,
          mt_log_probs,
          mt_targets,
          mt_input_lengths,
          mt_target_lengths,
          mt_loss,
          mt_alpha,
          InternalMemAlloc),
      "RunBwd");
  return result;
}

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
  return AT_DISPATCH_FLOATING_TYPES(
      log_probs.scalar_type(), "CtcLossBackward", [&] {
        return CtcLossBackwardImpl<scalar_t>(
            grad,
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            neg_log_likelihood,
            log_alpha,
            blank,
            zero_infinity);
      });
}

ADVANCED_REGISTER(aten, PrivateUse1, "_ctc_loss", CtcLoss)
ADVANCED_REGISTER(aten, PrivateUse1, "_ctc_loss_backward", CtcLossBackward)

} // namespace musa
} // namespace at
