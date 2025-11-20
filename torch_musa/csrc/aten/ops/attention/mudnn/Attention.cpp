#include <ATen/ExpandUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_scaled_dot_product_attention_flash_musa_backward_native.h>
#include <ATen/ops/_scaled_dot_product_attention_flash_musa_native.h>
#include <ATen/ops/_scaled_dot_product_attention_math_musa_backward_native.h>
#include <ATen/ops/_scaled_dot_product_attention_math_musa_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/sum_musa_dispatch.h>
#endif

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/attention/mudnn/SDPUtils.h"
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/aten/utils/Numeric.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at::musa {

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _FlashAttnForward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& cumulative_sequence_length_q,
    const std::optional<Tensor>& cumulative_sequence_length_k,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const std::optional<Tensor>& _seqused_k,
    const std::optional<Tensor>& _alibi_slopes) {
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3000)
  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
  std::optional<Tensor> out = std::nullopt;

  std::optional<Tensor> seqused_k = _seqused_k;
  std::optional<at::Tensor> block_table =
      std::nullopt; // we are not using the block table yet

  TORCH_CHECK(
      !_alibi_slopes.has_value(),
      "_flash_attention_forward doesn't support alibi_slopes for MUSA backend");
  TORCH_CHECK(
      !window_size_left.has_value(),
      "_flash_attention_forward doesn't support window_size_left for MUSA backend");
  TORCH_CHECK(
      !window_size_right.has_value(),
      "_flash_attention_forward doesn't support window_size_right for MUSA backend");

  // We are going to have two paths:
  // 1. The standard MHA path for dense tensors
  // 2. The Varseqlen path
  TORCH_CHECK(
      cumulative_sequence_length_q.has_value() ==
          cumulative_sequence_length_k.has_value(),
      "cumulative_sequence_length_q and cumulative_sequence_length_k must be both set or both not set");
  Tensor output, q_padded, k_padded, v_padded, logsumexp, output_shape,
      philox_seed, philox_offset, debug_attn_mask;
  if (cumulative_sequence_length_q.has_value()) {
    std::tie(output, logsumexp, debug_attn_mask) = MuDNNFlashVarlenFwd(
        query,
        key,
        value,
        cumulative_sequence_length_q.value(),
        cumulative_sequence_length_k.value(),
        max_seqlen_batch_q,
        max_seqlen_batch_k,
        dropout_p,
        softmax_scale,
        is_causal);
  } else {
    std::tie(output, logsumexp, debug_attn_mask) = MuDNNFlashSDPAFwd(
        query,
        key,
        value,
        /*attn_mask=*/std::nullopt,
        dropout_p,
        is_causal,
        softmax_scale);
  }
  debug_attn_mask =
      return_debug_mask ? debug_attn_mask : at::empty({0}, query.options());
  return std::make_tuple(
      std::move(output),
      std::move(logsumexp),
      std::move(philox_seed),
      std::move(philox_offset),
      std::move(debug_attn_mask));

#else
  TORCH_CHECK(false, "_flash_attention_forward was not enabled for build.")
  return std::make_tuple(Tensor(), Tensor(), Tensor(), Tensor(), Tensor());
#endif
}

// TODO(@ai-infra): dropout_p is not used in backward currently
std::tuple<Tensor, Tensor, Tensor> _FlashAttnBackward(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_k,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right) {
#if defined(MUDNN_VERSION) && (MUDNN_VERSION >= 3000)
  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
  //  CUDA code assumes that dout is contiguous
  auto contiguous_grad_out = grad_out.contiguous();
  auto contiguous_out = out.contiguous();

  TORCH_CHECK(
      !window_size_left.has_value(),
      "_flash_attention_forward doesn't support window_size_left for MUSA backend");
  TORCH_CHECK(
      !window_size_right.has_value(),
      "_flash_attention_forward doesn't support window_size_right for MUSA backend");

  at::Tensor dq;
  at::Tensor dk;
  at::Tensor dv;

  // We check the whether the cumulative_sequence_length_q is defined
  // in order to determine whether we are using varlen or dense backward
  if (cumulative_sequence_length_q.defined()) {
    // Varlen backward
    auto [dQuery, dKey, dValue, dSoftmax] = MuDNNFlashVarlenBwd(
        contiguous_grad_out,
        query,
        key,
        value,
        contiguous_out,
        logsumexp,
        dq,
        dk,
        dv,
        cumulative_sequence_length_q,
        cumulative_sequence_length_k,
        max_seqlen_batch_q,
        max_seqlen_batch_k,
        dropout_p,
        softmax_scale,
        is_causal);
    return std::make_tuple(
        std::move(dQuery), std::move(dKey), std::move(dValue));
  } else {
    // Dense backward
    auto [dQuery, dKey, dValue, dSoftmax] = MuDNNFlashSDPABwd(
        contiguous_grad_out,
        query,
        key,
        value,
        contiguous_out,
        logsumexp,
        /*dropout_mask=*/Tensor(),
        is_causal,
        /*attn_mask=*/Tensor(),
        softmax_scale);
    return std::make_tuple(
        std::move(dQuery), std::move(dKey), std::move(dValue));
  }
#else
  TORCH_CHECK(false, "USE_FLASH_ATTENTION was not enabled for build.");
  return std::make_tuple(Tensor(), Tensor(), Tensor());
#endif
}
} // namespace at::musa