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
#endif

#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at::musa {

namespace {

bool IsPadMask(
    const std::optional<Tensor>& mask,
    const Tensor& query,
    const Tensor& key) {
  return mask.has_value() && mask->defined() && mask->dim() == 2 &&
      mask->size(0) == query.size(0) && mask->size(1) == key.size(2);
}

void CheckScale(const Tensor& query, std::optional<double> scale) {
  using SFT = typename c10::SymFloat;

  if (scale.has_value()) {
    const auto default_scale = (SFT(1.0) / (SFT(query.sym_size(-1)).sqrt()));
    const auto user_scale = SFT(scale.value());
    TORCH_CHECK(
        default_scale == user_scale,
        "Now torch_musa only allows explicit value of `scale` parameter, ",
        "whose value is equal to `1/sqrt(query.size(-1))`.")
  }
}

} // anonymous namespace

// query shape: [N, H_q, L, E]
// key   shape: [N, H, S, E]
// value shape: [N, H, S, E_v]
std::tuple<Tensor, Tensor, Tensor> MuDNNMathSDPAFwd(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  auto mask = attn_mask; // remove const
  CheckScale(query, scale);
  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "Expect all query, key, value has 4D shape!");
  if (is_causal) {
    TORCH_CHECK(
        !attn_mask.has_value(),
        "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
    TORCH_CHECK(
        !query.is_nested() && !key.is_nested(),
        "_scaled_dot_product_attention: Nested tensors for query / key are not supported when is_causal=True");
    const auto L = query.sym_size(-2), S = key.sym_size(-2);
    mask = at::ones_symint({L, S}, query.options().dtype(at::kBool)).tril();
  }

  const c10::musa::MUSAGuard device_guard(query.device());
  const auto contig_q = query.contiguous();
  const auto contig_k = key.contiguous();
  const auto contig_v = value.contiguous();

  auto musa_q = CreateMUTensor(contig_q);
  auto musa_k = CreateMUTensor(contig_k);
  auto musa_v = CreateMUTensor(contig_v);

  const auto contig_mask = mask.has_value() ? mask.value().contiguous()
                                            : at::empty({0}, query.options());
  auto musa_mask = CreateMUTensor(contig_mask);

  const auto N = query.size(0);
  const auto H_q = query.size(1);
  const auto L = query.size(2);
  const auto E = query.size(3);

  const auto S = key.size(2);

  const auto E_v = value.size(3);

  auto contig_output = at::empty(
      {N, H_q, L, E_v}, query.options(), at::MemoryFormat::Contiguous);
  auto musa_out = CreateMUTensor(contig_output);

  auto contig_attn_weights =
      at::empty({N, H_q, L, S}, query.options(), at::MemoryFormat::Contiguous);
  auto musa_attn_weights = CreateMUTensor(contig_attn_weights);

  auto& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  CHECK_MUDNN_STATUS(
      sdpa.SetComputeMode(musa::GetComputeModeFromCtx(query.scalar_type())),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(H_q * E), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetMaskMode(IsPadMask(mask, query, key)), "SetMaskMode");

  const auto dropout_mask_opt = query.options().dtype(at::kBool);
  auto contig_dropout_mask = at::empty({0}, dropout_mask_opt);
  if (dropout_p > 0.0) {
    contig_dropout_mask = at::empty(
        {N, H_q, L, S}, dropout_mask_opt, at::MemoryFormat::Contiguous);

    sdpa.SetDropoutP(dropout_p);
    sdpa.SetTraining(true);
  }
  auto musa_dropout_mask = CreateMUTensor(contig_dropout_mask);

  CHECK_MUDNN_STATUS(
      sdpa.RunMath(
          h,
          musa_out,
          musa_attn_weights,
          musa_q,
          musa_k,
          musa_v,
          musa_mask,
          musa_dropout_mask,
          at::musa::InternalMemAlloc),
      "Run SDPA Math FWD.");

  return std::make_tuple(
      contig_output, contig_attn_weights, contig_dropout_mask);
}

std::tuple<Tensor, Tensor, Tensor> MuDNNMathSDPABwd(
    const Tensor& grad_output,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& output,
    const Tensor& attn_weights,
    const Tensor& dropout_mask) {
  const c10::musa::MUSAGuard device_guard(query.device());
  const auto contig_q = query.contiguous();
  const auto contig_k = key.contiguous();
  const auto contig_v = value.contiguous();

  auto musa_q = CreateMUTensor(contig_q);
  auto musa_k = CreateMUTensor(contig_k);
  auto musa_v = CreateMUTensor(contig_v);

  auto contig_grad_q = at::empty_like(contig_q, query.options());
  auto musa_grad_q = CreateMUTensor(contig_grad_q);

  auto contig_grad_k = at::empty_like(contig_k, key.options());
  auto musa_grad_k = CreateMUTensor(contig_grad_k);

  auto contig_grad_v = at::empty_like(contig_v, value.options());
  auto musa_grad_v = CreateMUTensor(contig_grad_v);

  const auto contig_grad_out = grad_output.contiguous();
  auto musa_grad_out = CreateMUTensor(contig_grad_out);

  const auto contig_attn_weights = attn_weights.contiguous();
  auto musa_attn_weights = CreateMUTensor(contig_attn_weights);

  auto contig_grad_attn_weights =
      at::empty_like(contig_attn_weights, attn_weights.options());
  auto musa_grad_attn_weights = CreateMUTensor(contig_grad_attn_weights);

  const auto contig_dropout_mask = dropout_mask.defined()
      ? dropout_mask.contiguous()
      : at::empty({0}, query.options().dtype(at::kBool));
  auto musa_dropout_mask = CreateMUTensor(contig_dropout_mask);

  auto& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  const auto H_q = query.size(1);
  const auto E = query.size(3);

  CHECK_MUDNN_STATUS(
      sdpa.SetComputeMode(musa::GetComputeModeFromCtx(query.scalar_type())),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(H_q * E), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetTraining(true), "SetTraining");

  CHECK_MUDNN_STATUS(
      sdpa.RunMathBwd(
          h,
          musa_grad_q,
          musa_grad_k,
          musa_grad_v,
          musa_grad_attn_weights,
          musa_grad_out,
          musa_q,
          musa_k,
          musa_v,
          musa_attn_weights,
          musa_dropout_mask,
          at::musa::InternalMemAlloc),
      "Run SDPA Math BWD.");

  return std::make_tuple(contig_grad_q, contig_grad_k, contig_grad_v);
}

std::tuple<Tensor, Tensor, Tensor> MuDNNFlashSDPAFwd(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  auto mask = attn_mask; // remove const
  CheckScale(query, scale);
  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "Expect all query, key, value has 4D shape!");
  if (is_causal) {
    TORCH_CHECK(
        !attn_mask.has_value(),
        "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
    TORCH_CHECK(
        !query.is_nested() && !key.is_nested(),
        "_scaled_dot_product_attention: Nested tensors for query / key are not supported when is_causal=True");
  }

  const c10::musa::MUSAGuard device_guard(query.device());

  const auto N = query.size(0);
  const auto H_q = query.size(1);
  const auto L = query.size(2);
  const auto E = query.size(3);

  const auto S = key.size(2);

  const auto E_v = value.size(3);

  auto musa_q = CreateMUTensor(query);
  auto musa_k = CreateMUTensor(key);
  auto musa_v = CreateMUTensor(value);

  auto output =
      at::empty({N, L, H_q, E_v}, query.options(), at::MemoryFormat::Contiguous)
          .transpose(1, 2);
  auto musa_out = CreateMUTensor(output);

  const auto contig_mask = mask.has_value() ? mask.value().contiguous()
                                            : at::empty({0}, query.options());
  auto musa_mask = CreateMUTensor(contig_mask);

  auto contig_logsumexp = at::empty(
      {N, H_q, L},
      query.options().dtype(at::kFloat),
      at::MemoryFormat::Contiguous);
  auto musa_lse = CreateMUTensor(contig_logsumexp);

  auto& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  CHECK_MUDNN_STATUS(sdpa.SetCausal(is_causal), "SetCausal");
  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(H_q * E), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetMaskMode(IsPadMask(mask, query, key)), "SetMaskMode");

  const auto dropout_mask_opt = query.options().dtype(at::kBool);
  auto contig_dropout_mask = at::empty({0}, dropout_mask_opt);
  if (dropout_p > 0.0) {
    contig_dropout_mask = at::empty(
        {N, H_q, L, S}, dropout_mask_opt, at::MemoryFormat::Contiguous);

    sdpa.SetDropoutP(dropout_p);
    sdpa.SetTraining(true);
  }
  auto musa_dropout_mask = CreateMUTensor(contig_dropout_mask);

  CHECK_MUDNN_STATUS(
      sdpa.RunFlash(
          h,
          musa_out,
          musa_lse,
          musa_q,
          musa_k,
          musa_v,
          musa_mask,
          musa_dropout_mask,
          at::musa::InternalMemAlloc),
      "Run SDPA Flash FWD.");

  return std::make_tuple(output, contig_logsumexp, contig_dropout_mask);
}

std::tuple<Tensor, Tensor, Tensor> MuDNNFlashSDPABwd(
    const Tensor& grad_output,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& output,
    const Tensor& logsumexp,
    const Tensor& dropout_mask,
    bool is_causal,
    const std::optional<Tensor>& attn_mask) {
  const c10::musa::MUSAGuard device_guard(query.device());

  auto musa_q = CreateMUTensor(query);
  auto musa_k = CreateMUTensor(key);
  auto musa_v = CreateMUTensor(value);

  auto grad_q = at::empty_like(query, query.options());
  auto musa_grad_q = CreateMUTensor(grad_q);

  auto grad_k = at::empty_like(key, key.options());
  auto musa_grad_k = CreateMUTensor(grad_k);

  auto grad_v = at::empty_like(value, value.options());
  auto musa_grad_v = CreateMUTensor(grad_v);

  const auto reformatted_grad_output = ContiguousIfZeroInStrides(grad_output);
  auto musa_grad_output = CreateMUTensor(reformatted_grad_output);

  const auto contig_logsumexp = logsumexp.contiguous();
  auto musa_logsumexp = CreateMUTensor(contig_logsumexp);

  const auto contig_dropout_mask = dropout_mask.contiguous();
  auto musa_dropout_mask = CreateMUTensor(contig_dropout_mask);

  auto musa_output = CreateMUTensor(output);

  auto& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  auto contig_mask = (attn_mask.has_value() && attn_mask->defined())
      ? attn_mask.value().contiguous()
      : at::empty({0}, query.options());

  CHECK_MUDNN_STATUS(sdpa.SetMaskMode(IsPadMask(attn_mask, query, key)), "SetMaskMode");
  auto musa_mask = CreateMUTensor(contig_mask);

  const auto H_q = query.size(1);
  const auto E = query.size(3);

  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(H_q * E), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetTraining(true), "SetTraining");
  CHECK_MUDNN_STATUS(sdpa.SetCausal(is_causal), "SetCausal");

  CHECK_MUDNN_STATUS(
      sdpa.RunFlashBwd(
          h,
          musa_grad_q,
          musa_grad_k,
          musa_grad_v,
          musa_grad_output,
          musa_q,
          musa_k,
          musa_v,
          musa_mask,
          musa_output,
          musa_logsumexp,
          musa_dropout_mask,
          at::musa::InternalMemAlloc),
      "Run SDPA Flash BWD.");

  return std::make_tuple(grad_q, grad_k, grad_v);
}

} // namespace at::musa
