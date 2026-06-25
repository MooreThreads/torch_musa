#include "torch_musa/csrc/aten/mudnn/ScaledDotProductAttention.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

double ScaledDotProductAttention::GetScale() const {
  return has_scale_ ? scale_ : 0.0;
}

void ScaledDotProductAttention::BuildStandardDescriptor() const {
  if (dropout_p_ > 0.0) {
    CHECK_MUDNN_STATUS(
        mudnnSetScaledDotProductAttentionDescriptorEx(
            Desc(),
            compute_mode_,
            embed_dim_,
            batch_size_,
            num_heads_,
            GetScale(),
            use_pad_mask_,
            /*keyFormatBHDS=*/false,
            is_training_,
            is_causal_,
            /*isDeterministic=*/false,
            dropout_p_,
            seed_,
            offset_,
            dropout_mode_,
            const_cast<uint64_t*>(device_seed_),
            const_cast<uint64_t*>(device_offset_),
            graph_offset_),
        "mudnnSetScaledDotProductAttentionDescriptorEx");
  } else {
    CHECK_MUDNN_STATUS(
        mudnnSetScaledDotProductAttentionDescriptor(
            Desc(),
            compute_mode_,
            embed_dim_,
            batch_size_,
            num_heads_,
            GetScale(),
            use_pad_mask_,
            /*keyFormatBHDS=*/false,
            is_training_,
            is_causal_,
            /*isDeterministic=*/false),
        "mudnnSetScaledDotProductAttentionDescriptor");
  }
}

void ScaledDotProductAttention::BuildVarlenDescriptor() const {
  if (dropout_p_ > 0.0) {
    CHECK_MUDNN_STATUS(
        mudnnSetScaledDotProductAttentionVarlenDescriptorEx(
            Desc(),
            compute_mode_,
            embed_dim_,
            batch_size_,
            num_heads_,
            GetScale(),
            use_pad_mask_,
            is_training_,
            is_causal_,
            /*isDeterministic=*/false,
            max_seqlen_q_,
            max_seqlen_k_,
            dropout_p_,
            seed_,
            offset_,
            dropout_mode_,
            const_cast<uint64_t*>(device_seed_),
            const_cast<uint64_t*>(device_offset_),
            graph_offset_),
        "mudnnSetScaledDotProductAttentionVarlenDescriptorEx");
  } else {
    CHECK_MUDNN_STATUS(
        mudnnSetScaledDotProductAttentionVarlenDescriptor(
            Desc(),
            compute_mode_,
            embed_dim_,
            batch_size_,
            num_heads_,
            GetScale(),
            use_pad_mask_,
            is_training_,
            is_causal_,
            /*isDeterministic=*/false,
            max_seqlen_q_,
            max_seqlen_k_),
        "mudnnSetScaledDotProductAttentionVarlenDescriptor");
  }
}

// ---- Math forward ----

mudnnStatus_t ScaledDotProductAttention::RunMath(
    muHandle& h,
    muTensor& out,
    muTensor& attn_weights,
    const muTensor& q,
    const muTensor& k,
    const muTensor& v,
    const muTensor& mask,
    muTensor& dropout_mask,
    const MemoryMaintainer& maintainer) const {
  BuildStandardDescriptor();
  out.Build();
  attn_weights.Build();
  q.Build();
  k.Build();
  v.Build();
  mask.Build();
  dropout_mask.Build();

  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnScaledDotProductAttentionMathForwardGetWorkspaceSize(
          h,
          Desc(),
          attn_weights.Desc(),
          q.Desc(),
          k.Desc(),
          v.Desc(),
          mask.Desc(),
          mask.DataPtr(),
          dropout_mask.Desc(),
          out.Desc(),
          &ws_size),
      "mudnnScaledDotProductAttentionMathForwardGetWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnScaledDotProductAttentionMathForward(
      h,
      MUDNN_UNPACK_TENSOR(attn_weights),
      MUDNN_UNPACK_TENSOR(q),
      MUDNN_UNPACK_TENSOR(k),
      MUDNN_UNPACK_TENSOR(v),
      MUDNN_UNPACK_TENSOR(mask),
      Desc(),
      dropout_mask.Desc(),
      dropout_mask.DataPtr(),
      out.Desc(),
      out.DataPtr(),
      mem.get(),
      ws_size);
}

// ---- Math backward ----

mudnnStatus_t ScaledDotProductAttention::RunMathBwd(
    muHandle& h,
    muTensor& grad_q,
    muTensor& grad_k,
    muTensor& grad_v,
    muTensor& grad_attn_weights,
    const muTensor& grad_out,
    const muTensor& q,
    const muTensor& k,
    const muTensor& v,
    const muTensor& attn_weights,
    const muTensor& dropout_mask,
    const MemoryMaintainer& maintainer) const {
  BuildStandardDescriptor();
  grad_q.Build();
  grad_k.Build();
  grad_v.Build();
  grad_attn_weights.Build();
  grad_out.Build();
  q.Build();
  k.Build();
  v.Build();
  attn_weights.Build();
  dropout_mask.Build();

  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnScaledDotProductAttentionMathBackwardGetWorkspaceSize(
          h,
          Desc(),
          grad_out.Desc(),
          q.Desc(),
          k.Desc(),
          v.Desc(),
          dropout_mask.Desc(),
          &ws_size),
      "mudnnScaledDotProductAttentionMathBackwardGetWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnScaledDotProductAttentionMathBackward(
      h,
      MUDNN_UNPACK_TENSOR(grad_attn_weights),
      MUDNN_UNPACK_TENSOR(grad_out),
      MUDNN_UNPACK_TENSOR(q),
      MUDNN_UNPACK_TENSOR(k),
      MUDNN_UNPACK_TENSOR(v),
      MUDNN_UNPACK_TENSOR(attn_weights),
      MUDNN_UNPACK_TENSOR(dropout_mask),
      Desc(),
      grad_q.Desc(),
      grad_q.DataPtr(),
      grad_k.Desc(),
      grad_k.DataPtr(),
      grad_v.Desc(),
      grad_v.DataPtr(),
      mem.get(),
      ws_size);
}

// ---- Flash forward ----

mudnnStatus_t ScaledDotProductAttention::RunFlash(
    muHandle& h,
    muTensor& out,
    muTensor& lse,
    const muTensor& q,
    const muTensor& k,
    const muTensor& v,
    const muTensor& mask,
    muTensor& dropout_mask,
    const MemoryMaintainer& /*maintainer*/) const {
  BuildStandardDescriptor();
  out.Build();
  lse.Build();
  q.Build();
  k.Build();
  v.Build();
  mask.Build();
  dropout_mask.Build();

  return mudnnScaledDotProductAttentionFlashForward(
      h,
      MUDNN_UNPACK_TENSOR(lse),
      MUDNN_UNPACK_TENSOR(q),
      MUDNN_UNPACK_TENSOR(k),
      MUDNN_UNPACK_TENSOR(v),
      MUDNN_UNPACK_TENSOR(mask),
      Desc(),
      dropout_mask.Desc(),
      dropout_mask.DataPtr(),
      out.Desc(),
      out.DataPtr());
}

// ---- Flash backward ----

mudnnStatus_t ScaledDotProductAttention::RunFlashBwd(
    muHandle& h,
    muTensor& grad_q,
    muTensor& grad_k,
    muTensor& grad_v,
    const muTensor& grad_out,
    const muTensor& q,
    const muTensor& k,
    const muTensor& v,
    const muTensor& mask,
    const muTensor& out,
    const muTensor& lse,
    const muTensor& dropout_mask,
    const MemoryMaintainer& maintainer) const {
  BuildStandardDescriptor();
  grad_q.Build();
  grad_k.Build();
  grad_v.Build();
  grad_out.Build();
  q.Build();
  k.Build();
  v.Build();
  mask.Build();
  out.Build();
  lse.Build();
  dropout_mask.Build();

  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnScaledDotProductAttentionFlashBackwardGetWorkspaceSize(
          h,
          Desc(),
          grad_out.Desc(),
          q.Desc(),
          k.Desc(),
          v.Desc(),
          mask.Desc(),
          out.Desc(),
          lse.Desc(),
          dropout_mask.Desc(),
          grad_q.Desc(),
          grad_k.Desc(),
          grad_v.Desc(),
          &ws_size),
      "mudnnScaledDotProductAttentionFlashBackwardGetWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnScaledDotProductAttentionFlashBackward(
      h,
      MUDNN_UNPACK_TENSOR(grad_out),
      MUDNN_UNPACK_TENSOR(q),
      MUDNN_UNPACK_TENSOR(k),
      MUDNN_UNPACK_TENSOR(v),
      MUDNN_UNPACK_TENSOR(mask),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(lse),
      MUDNN_UNPACK_TENSOR(dropout_mask),
      Desc(),
      grad_q.Desc(),
      grad_q.DataPtr(),
      grad_k.Desc(),
      grad_k.DataPtr(),
      grad_v.Desc(),
      grad_v.DataPtr(),
      mem.get(),
      ws_size);
}

// ---- Flash Varlen forward ----

mudnnStatus_t ScaledDotProductAttention::RunFlashVarlen(
    muHandle& h,
    muTensor& out,
    muTensor& lse,
    const muTensor& q,
    const muTensor& k,
    const muTensor& v,
    const muTensor& mask,
    muTensor& dropout_mask,
    const muTensor& cu_seqlens_q,
    const muTensor& cu_seqlens_k,
    const MemoryMaintainer& /*maintainer*/) const {
  BuildVarlenDescriptor();
  out.Build();
  lse.Build();
  q.Build();
  k.Build();
  v.Build();
  mask.Build();
  dropout_mask.Build();
  cu_seqlens_q.Build();
  cu_seqlens_k.Build();

  return mudnnScaledDotProductAttentionFlashVarlenForward(
      h,
      MUDNN_UNPACK_TENSOR(lse),
      MUDNN_UNPACK_TENSOR(q),
      MUDNN_UNPACK_TENSOR(k),
      MUDNN_UNPACK_TENSOR(v),
      MUDNN_UNPACK_TENSOR(mask),
      MUDNN_UNPACK_TENSOR(cu_seqlens_q),
      MUDNN_UNPACK_TENSOR(cu_seqlens_k),
      Desc(),
      dropout_mask.Desc(),
      dropout_mask.DataPtr(),
      out.Desc(),
      out.DataPtr());
}

// ---- Flash Varlen backward ----

mudnnStatus_t ScaledDotProductAttention::RunFlashVarlenBwd(
    muHandle& h,
    muTensor& grad_q,
    muTensor& grad_k,
    muTensor& grad_v,
    const muTensor& grad_out,
    const muTensor& q,
    const muTensor& k,
    const muTensor& v,
    const muTensor& mask,
    const muTensor& out,
    const muTensor& lse,
    const muTensor& dropout_mask,
    const muTensor& cu_seqlens_q,
    const muTensor& cu_seqlens_k,
    const MemoryMaintainer& maintainer) const {
  BuildVarlenDescriptor();
  grad_q.Build();
  grad_k.Build();
  grad_v.Build();
  grad_out.Build();
  q.Build();
  k.Build();
  v.Build();
  mask.Build();
  out.Build();
  lse.Build();
  dropout_mask.Build();
  cu_seqlens_q.Build();
  cu_seqlens_k.Build();

  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnScaledDotProductAttentionFlashVarlenBackwardGetWorkspaceSize(
          h,
          Desc(),
          grad_out.Desc(),
          q.Desc(),
          k.Desc(),
          v.Desc(),
          mask.Desc(),
          out.Desc(),
          lse.Desc(),
          dropout_mask.Desc(),
          cu_seqlens_q.Desc(),
          cu_seqlens_k.Desc(),
          grad_q.Desc(),
          grad_k.Desc(),
          grad_v.Desc(),
          &ws_size),
      "mudnnScaledDotProductAttentionFlashVarlenBackwardGetWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnScaledDotProductAttentionFlashVarlenBackward(
      h,
      MUDNN_UNPACK_TENSOR(grad_out),
      MUDNN_UNPACK_TENSOR(q),
      MUDNN_UNPACK_TENSOR(k),
      MUDNN_UNPACK_TENSOR(v),
      MUDNN_UNPACK_TENSOR(mask),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(lse),
      MUDNN_UNPACK_TENSOR(dropout_mask),
      MUDNN_UNPACK_TENSOR(cu_seqlens_q),
      MUDNN_UNPACK_TENSOR(cu_seqlens_k),
      Desc(),
      grad_q.Desc(),
      grad_q.DataPtr(),
      grad_k.Desc(),
      grad_k.DataPtr(),
      grad_v.Desc(),
      grad_v.DataPtr(),
      mem.get(),
      ws_size);
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // namespace at::musa
