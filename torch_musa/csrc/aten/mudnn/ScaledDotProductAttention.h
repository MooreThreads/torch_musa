#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_SCALEDDOTPRODUCTATTENTION_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_SCALEDDOTPRODUCTATTENTION_H_

#include <cstdint>

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Convolution.h"
#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class ScaledDotProductAttention
    : public Descriptor<
          mudnnScaledDotProductAttentionStruct,
          mudnnCreateScaledDotProductAttentionDescriptor,
          mudnnDestroyScaledDotProductAttentionDescriptor> {
 public:
  mudnnStatus_t SetComputeMode(ComputeMode mode) {
    compute_mode_ = mode;
    return MUDNN_STATUS_SUCCESS;
  }
  mudnnStatus_t SetEmbedDim(int dim) {
    embed_dim_ = dim;
    return MUDNN_STATUS_SUCCESS;
  }
  mudnnStatus_t SetBatchSize(int bs) {
    batch_size_ = bs;
    return MUDNN_STATUS_SUCCESS;
  }
  mudnnStatus_t SetHeadsNum(int n) {
    num_heads_ = n;
    return MUDNN_STATUS_SUCCESS;
  }
  mudnnStatus_t SetScale(double s) {
    scale_ = s;
    has_scale_ = true;
    return MUDNN_STATUS_SUCCESS;
  }
  mudnnStatus_t SetMaskMode(bool pad_mask) {
    use_pad_mask_ = pad_mask;
    return MUDNN_STATUS_SUCCESS;
  }
  mudnnStatus_t SetTraining(bool t) {
    is_training_ = t;
    return MUDNN_STATUS_SUCCESS;
  }
  mudnnStatus_t SetCausal(bool c) {
    is_causal_ = c;
    return MUDNN_STATUS_SUCCESS;
  }
  mudnnStatus_t SetDropoutP(double p) {
    dropout_p_ = p;
    return MUDNN_STATUS_SUCCESS;
  }
  mudnnStatus_t SetSeed(
      uint64_t seed,
      uint64_t offset,
      int dropout_mode,
      const uint64_t* device_seed,
      const uint64_t* device_offset,
      uint64_t graph_offset) {
    seed_ = seed;
    offset_ = offset;
    dropout_mode_ = dropout_mode;
    device_seed_ = device_seed;
    device_offset_ = device_offset;
    graph_offset_ = graph_offset;
    return MUDNN_STATUS_SUCCESS;
  }
  mudnnStatus_t SetMaxSeqlenQ(int len) {
    max_seqlen_q_ = len;
    return MUDNN_STATUS_SUCCESS;
  }
  mudnnStatus_t SetMaxSeqlenK(int len) {
    max_seqlen_k_ = len;
    return MUDNN_STATUS_SUCCESS;
  }

  mudnnStatus_t RunMath(
      muHandle& h,
      muTensor& out,
      muTensor& attn_weights,
      const muTensor& q,
      const muTensor& k,
      const muTensor& v,
      const muTensor& mask,
      muTensor& dropout_mask,
      const MemoryMaintainer& m) const;

  mudnnStatus_t RunMathBwd(
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
      const MemoryMaintainer& m) const;

  mudnnStatus_t RunFlash(
      muHandle& h,
      muTensor& out,
      muTensor& lse,
      const muTensor& q,
      const muTensor& k,
      const muTensor& v,
      const muTensor& mask,
      muTensor& dropout_mask,
      const MemoryMaintainer& m) const;

  mudnnStatus_t RunFlashBwd(
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
      const MemoryMaintainer& m) const;

  mudnnStatus_t RunFlashVarlen(
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
      const MemoryMaintainer& m) const;

  mudnnStatus_t RunFlashVarlenBwd(
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
      const MemoryMaintainer& m) const;

 private:
  void BuildStandardDescriptor() const;
  void BuildVarlenDescriptor() const;
  double GetScale() const;
  mudnnMathType_t GetMathType() const;

  ComputeMode compute_mode_ = MUDNN_DEFAULT_MATH;
  int embed_dim_ = 0;
  int batch_size_ = 0;
  int num_heads_ = 0;
  double scale_ = 0.0;
  bool has_scale_ = false;
  bool use_pad_mask_ = false;
  bool is_training_ = false;
  bool is_causal_ = false;
  double dropout_p_ = 0.0;
  uint64_t seed_ = 0;
  uint64_t offset_ = 0;
  int dropout_mode_ = 0;
  const uint64_t* device_seed_ = nullptr;
  const uint64_t* device_offset_ = nullptr;
  uint64_t graph_offset_ = 0;
  int max_seqlen_q_ = 0;
  int max_seqlen_k_ = 0;
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::ScaledDotProductAttention;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::ScaledDotProductAttention;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API
#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_SCALEDDOTPRODUCTATTENTION_H_
