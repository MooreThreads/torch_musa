#include <type_traits>

#include <mudnn.h>

#include <ATen/NativeFunctions.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/transformers/attention.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/ops/attention/mudnn/SDPUtils.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace musa {

inline bool is_pad_mask(const at::Tensor& mask, const at::Tensor& query) {
  return mask.dim() == 2 && mask.size(0) == query.size(0) &&
      mask.size(1) == query.size(2);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> MuDNNMathSDPAFwd(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal) {
  auto mask = attn_mask; // remove const
  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "Expect all query, key, value has 4D shape!");
  TORCH_CHECK(
      query.size(1) == key.size(1),
      "Expect query's sequence length same as key's sequence length!");
  if (is_causal) {
    TORCH_CHECK(
        !attn_mask.has_value(),
        "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
    TORCH_CHECK(
        !query.is_nested() && !key.is_nested(),
        "_scaled_dot_product_attention: Nested tensors for query / key are not supported when is_causal=True");
    const auto q_seq_len = query.sym_size(-2), kv_seq_len = key.sym_size(-2);
    mask = at::ones_symint(
               {q_seq_len, kv_seq_len}, query.options().dtype(at::kBool))
               .tril();
  }

  c10::musa::MUSAGuard device_guard(query.device());
  auto contiguous_query = query.contiguous();
  auto contiguous_key = key.contiguous();
  auto contiguous_value = value.contiguous();

  auto musa_q = at::musa::CreateMUTensor(contiguous_query);
  auto musa_k = at::musa::CreateMUTensor(contiguous_key);
  auto musa_v = at::musa::CreateMUTensor(contiguous_value);

  // check the mask
  auto contiguous_mask = at::empty({0}); // should we keep this tensor in host?
  if (mask.has_value()) {
    contiguous_mask = mask.value().contiguous();
  }
  auto musa_mask = at::musa::CreateMUTensor(contiguous_mask);

  // Query (Batch x Num_heads x Q_seq_len  x Dim_per_head)
  // Key   (Batch x Num_heads x KV_seq_len x Dim_per_head)
  // Value (Batch x Num_heads x KV_seq_len x Dim_per_head)
  auto head_dim = query.sizes()[3]; // head_dim
  auto q_seq_len = query.size(2); // seq_len
  auto head_num = query.sizes()[1]; // head_num
  auto batch_size = query.sizes()[0]; // batch_size
  int32_t kv_seq_len; // kv_seq_len
  if (key.size(3) == head_dim) {
    kv_seq_len = key.size(2);
  } else {
    // key has shape [bs, head_num, head_dim, kv_seq_len]
    kv_seq_len = key.size(3);
  }

  auto output = at::empty(
      {batch_size, head_num, q_seq_len, head_dim},
      query.options(),
      at::MemoryFormat::Contiguous);
  auto musa_out = at::musa::CreateMUTensor(output);

  auto atten_probs = at::empty(
      {batch_size, head_num, q_seq_len, kv_seq_len},
      query.options(),
      at::MemoryFormat::Contiguous);
  auto musa_atten_probs = at::musa::CreateMUTensor(atten_probs);

  musa::muHandle& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  // Config Mudnn
  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(head_dim * head_num), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(head_num), "SetHeadsNum");
  if (mask.has_value()) {
    CHECK_MUDNN_STATUS(
        sdpa.SetMaskMode(is_pad_mask(mask.value(), query)), "SetMaskMode");
  }

  // store dropout mask, bool data type
  auto dropout_mask =
      at::empty({0}, query.options(), at::MemoryFormat::Contiguous);
  if (dropout_p > 0.0) {
    dropout_mask = at::empty(
        {batch_size, head_num, q_seq_len, kv_seq_len},
        query.options().dtype(at::kBool),
        at::MemoryFormat::Contiguous);

    sdpa.SetDropoutP(dropout_p);
    sdpa.SetTraining(true);
  }

  auto musa_dropout_mask = at::musa::CreateMUTensor(dropout_mask);

  CHECK_MUDNN_STATUS(
      sdpa.RunMath(
          h,
          musa_out,
          musa_atten_probs,
          musa_q,
          musa_k,
          musa_v,
          musa_mask,
          musa_dropout_mask,
          at::musa::InternalMemAlloc),
      "Run SDPA");

  return std::make_tuple(output, atten_probs, dropout_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> MuDNNMathSDPABwd(
    const at::Tensor& grad_output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& output,
    const at::Tensor& attn_weights,
    const at::Tensor& dropout_mask) {
  c10::musa::MUSAGuard device_guard(query.device());
  auto contiguous_query = query.contiguous();
  auto contiguous_key = key.contiguous();
  auto contiguous_value = value.contiguous();

  auto musa_q = at::musa::CreateMUTensor(contiguous_query);
  auto musa_k = at::musa::CreateMUTensor(contiguous_key);
  auto musa_v = at::musa::CreateMUTensor(contiguous_value);

  auto grad_query =
      at::empty_like(query, query.options(), at::MemoryFormat::Contiguous);
  auto musa_grad_query = at::musa::CreateMUTensor(grad_query);

  auto grad_key =
      at::empty_like(key, key.options(), at::MemoryFormat::Contiguous);
  auto musa_grad_key = at::musa::CreateMUTensor(grad_key);

  auto grad_value =
      at::empty_like(value, value.options(), at::MemoryFormat::Contiguous);
  auto musa_grad_value = at::musa::CreateMUTensor(grad_value);

  auto conti_grad_output = grad_output.contiguous();
  auto musa_grad_output = at::musa::CreateMUTensor(conti_grad_output);
  auto conti_attn_weights = attn_weights.contiguous();
  auto musa_attn_weights = at::musa::CreateMUTensor(conti_attn_weights);

  auto conti_dropout_masks = dropout_mask;
  if (dropout_mask.defined()) {
    conti_dropout_masks = dropout_mask.contiguous();
  }
  auto musa_dropout_mask = at::musa::CreateMUTensor(conti_dropout_masks);

  musa::muHandle& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  // query: [batch_size,head_num, Q_seq_len, head_dim]
  // key: [batch_size, head_num, KV_seq_len, head_dim]
  // val: [batch_size, head_num, KV_seq_len, head_dim]
  auto head_dim = query.sizes()[3]; // head_dim
  auto q_seq_len = query.sizes()[2]; // seq_len
  auto head_num = query.sizes()[1]; // head_num
  auto batch_size = query.sizes()[0]; // batch_size
  auto kv_seq_len = key.size(2);

  // batchfirst doesn't takes effect in SDPA actually.
  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(head_num * head_dim), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(head_num), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetTraining(true), "SetTraining");

  auto grad_attn_weights = at::empty_like(
      attn_weights, attn_weights.options(), at::MemoryFormat::Contiguous);
  auto musa_grad_attn_weights = at::musa::CreateMUTensor(grad_attn_weights);
  CHECK_MUDNN_STATUS(
      sdpa.RunMathBwd(
          h,
          musa_grad_query,
          musa_grad_key,
          musa_grad_value,
          musa_grad_attn_weights,
          musa_grad_output,
          musa_q,
          musa_k,
          musa_v,
          musa_attn_weights,
          musa_dropout_mask,
          at::musa::InternalMemAlloc),
      "Run SDPA bwd");

  return std::make_tuple(grad_query, grad_key, grad_value);
}

std::tuple<at::Tensor, at::Tensor> MuDNNFlashSDPAFwd(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal) {
  auto mask = attn_mask; // remove const
  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "Expect all query, key, value has 4D shape!");
  TORCH_CHECK(
      query.size(1) == key.size(1),
      "Expect query's sequence length same as key's sequence length!");
  if (is_causal) {
    TORCH_CHECK(
        !attn_mask.has_value(),
        "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
    TORCH_CHECK(
        !query.is_nested() && !key.is_nested(),
        "_scaled_dot_product_attention: Nested tensors for query / key are not supported when is_causal=True");
    const auto q_seq_len = query.sym_size(-2), kv_seq_len = key.sym_size(-2);
    mask = at::ones_symint(
               {q_seq_len, kv_seq_len}, query.options().dtype(at::kBool))
               .tril();
  }

  c10::musa::MUSAGuard device_guard(query.device());

  // Query (Batch x Num_heads x Q_seq_len  x Dim_per_head)
  // Key   (Batch x Num_heads x KV_seq_len x Dim_per_head)
  // Value (Batch x Num_heads x KV_seq_len x Dim_per_head)
  auto head_dim = query.sizes()[3]; // head_dim
  auto q_seq_len = query.size(2); // seq_len
  auto head_num = query.sizes()[1]; // head_num
  auto batch_size = query.sizes()[0]; // batch_size

  at::Tensor contiguous_query, contiguous_key, contiguous_value;
  musa::muTensor musa_q, musa_k, musa_v;

  at::Tensor output;
  musa::muTensor musa_out;
  musa_q = at::musa::CreateMUTensor(query);
  musa_k = at::musa::CreateMUTensor(key);
  musa_v = at::musa::CreateMUTensor(value);

  output = at::empty(
      {batch_size, q_seq_len, head_num, head_dim},
      query.options(),
      at::MemoryFormat::Contiguous);
  output = at::transpose(output, 1, 2);
  musa_out = at::musa::CreateMUTensor(output);

  // check the mask
  auto contiguous_mask = at::empty({0}); // should we keep this tensor in host?
  if (mask.has_value()) {
    contiguous_mask = mask.value().contiguous();
  }
  auto musa_mask = at::musa::CreateMUTensor(contiguous_mask);

  int32_t kv_seq_len; // kv_seq_len
  if (key.size(3) == head_dim) {
    kv_seq_len = key.size(2);
  } else {
    // key has shape [bs, head_num, head_dim, kv_seq_len]
    kv_seq_len = key.size(3);
  }

  auto log_sum_exp = at::empty(
      {batch_size, head_num, q_seq_len},
      query.options().dtype(at::kFloat),
      at::MemoryFormat::Contiguous);
  auto musa_lse = at::musa::CreateMUTensor(log_sum_exp);

  musa::muHandle& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  // Config Mudnn
  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(head_dim * head_num), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(head_num), "SetHeadsNum");
  if (mask.has_value()) {
    CHECK_MUDNN_STATUS(
        sdpa.SetMaskMode(is_pad_mask(mask.value(), query)), "SetMaskMode");
  }

  // store dropout mask, bool data type
  auto dropout_mask =
      at::empty({0}, query.options(), at::MemoryFormat::Contiguous);
  if (dropout_p > 0.0) {
    dropout_mask = at::empty(
        {batch_size, head_num, q_seq_len, kv_seq_len},
        query.options().dtype(at::kBool),
        at::MemoryFormat::Contiguous);

    sdpa.SetDropoutP(dropout_p);
    sdpa.SetTraining(true);
  }

  auto musa_dropout_mask = at::musa::CreateMUTensor(dropout_mask);

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
      "Run SDPA");

  return std::make_tuple(output, dropout_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> MuDNNFlashSDPABwd(
    const at::Tensor& grad_output,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& output,
    const at::Tensor& dropout_mask){TORCH_CHECK(
    false,
    "FlashAttention in MUSA backned doesn't support backward now. \nPlease disable the flash attention with `torch.backends.cuda.sdp_kernel(enable_flash=False)` if you want to train model temporarily.")}

ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_scaled_dot_product_attention_flash_musa",
    MuDNNFlashSDPAFwd);

ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_scaled_dot_product_attention_flash_musa_backward",
    MuDNNFlashSDPABwd);

ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_scaled_dot_product_attention_math_musa",
    MuDNNMathSDPAFwd);

ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_scaled_dot_product_attention_math_musa_backward",
    MuDNNMathSDPABwd);

} // namespace musa
} // namespace at
