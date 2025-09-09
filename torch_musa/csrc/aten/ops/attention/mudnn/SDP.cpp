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
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/aten/utils/Numeric.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at::musa {

namespace {

// Since MUSA not perform the `convert_boolean_attn_mask` preprocessing work,
// floating/boolean masks will be distinguished as different types.
// clang-format off
enum class MaskType {
  NONE = 0,
  CAUSAL = 1,
  // Assume that full mask shape is [N, H, L, S]
  // Floating masks
  FLT_LS         = 100, // 2D
  FLT_NS         = 101, // key padding
  FLT_HLS_BCAST  = 102, // 3-dim no batch broadcast
  FLT_HLS_FULL   = 103, // 3-dim no batch no broadcast
  FLT_NHLS_BCAST = 104, // 4-dim broadcast
  FLT_NHLS_FULL  = 105, // no broadcast
  // Binary masks
  BIN_LS         = 200, // 2D
  BIN_NS         = 201, // key padding
  BIN_HLS_BCAST  = 202, // 3-dim no batch broadcast
  BIN_HLS_FULL   = 203, // 3-dim no batch no broadcast
  BIN_NHLS_BCAST = 204, // 4-dim broadcast
  BIN_NHLS_FULL  = 205, // no broadcast
};
// clang-format on

MaskType ParseMaskType(
    const std::optional<Tensor>& mask,
    bool is_causal,
    int64_t N,
    int64_t H_q,
    int64_t L,
    int64_t S) {
  if (is_causal) {
    return MaskType::CAUSAL;
  }
  if (!mask.has_value() || !mask->defined()) {
    return MaskType::NONE;
  }

  const auto ndim = mask->dim();
  const auto shape = mask->sizes();
  const auto is_binary_mask = (mask->scalar_type() == ScalarType::Bool);

  if (ndim == 2) {
    if (shape[0] == L && shape[1] == S) {
      return is_binary_mask ? MaskType::BIN_LS : MaskType::FLT_LS;
    }
    if (shape[0] == N && shape[1] == S) {
      return is_binary_mask ? MaskType::BIN_NS : MaskType::FLT_NS;
    }
    TORCH_CHECK(
        false, "MUSA SPDA: shape of 2D attn_mask should be [L, S] or [N, S].");
  }

  if (ndim == 3) {
    if (shape == IntArrayRef({H_q, L, S})) {
      return is_binary_mask ? MaskType::BIN_HLS_FULL : MaskType::FLT_HLS_FULL;
    }
    TORCH_CHECK(
        at::is_expandable_to(shape, {H_q, L, S}),
        "MUSA SPDA: shape of 3D attn_mask should be expandable to [H_q, L, S].");
    return is_binary_mask ? MaskType::BIN_HLS_BCAST : MaskType::FLT_HLS_BCAST;
  }

  if (ndim == 4) {
    if (shape == IntArrayRef({N, H_q, L, S})) {
      return is_binary_mask ? MaskType::BIN_NHLS_FULL : MaskType::FLT_NHLS_FULL;
    }
    TORCH_CHECK(
        at::is_expandable_to(shape, {N, H_q, L, S}),
        "MUSA SPDA: ",
        "shape of 4D attn_mask should be expandable to [N, H_q, L, S].");
    return is_binary_mask ? MaskType::BIN_NHLS_BCAST : MaskType::FLT_NHLS_BCAST;
  }

  TORCH_CHECK(false, "MUSA SPDA: attn_mask should be 2/3/4D Tensor.");
}

bool HasMask(MaskType m) noexcept {
  return (m != MaskType::NONE) && (m != MaskType::CAUSAL);
}

bool IsPadMask(MaskType m) noexcept {
  return (m == MaskType::FLT_NS) || (m == MaskType::BIN_NS);
}

// dim-3 no batch, needs unsqueeze/squeeze
bool IsNoBatchMask(MaskType m) noexcept {
  return (m == MaskType::BIN_HLS_FULL) || (m == MaskType::FLT_HLS_FULL) ||
      (m == MaskType::BIN_HLS_BCAST) || (m == MaskType::FLT_HLS_BCAST);
}

bool IsValidGradMask(MaskType m) noexcept {
  const auto val = static_cast<int64_t>(m);
  constexpr auto low = static_cast<int64_t>(MaskType::FLT_LS);
  constexpr auto high = static_cast<int64_t>(MaskType::FLT_NHLS_FULL);
  return (val >= low) && (val <= high);
}

Tensor CalMaskGrad(
    const Tensor& contig_grad_mask_full,
    const Tensor& mask,
    MaskType mask_type) {
  const auto full_sizes = contig_grad_mask_full.sizes();
  const auto N = full_sizes[0];
  const auto H_q = full_sizes[1];
  const auto L = full_sizes[2];
  const auto S = full_sizes[3];
  switch (mask_type) {
    case MaskType::FLT_LS: {
      Tensor grad = mask.new_empty({L, S});
      at::musa::sum_out(grad, contig_grad_mask_full, {0, 1});
      return grad;
    }
    case MaskType::FLT_NS: {
      Tensor grad = mask.new_empty({N, S});
      at::musa::sum_out(grad, contig_grad_mask_full, {1, 2});
      return grad;
    }
    case MaskType::FLT_HLS_FULL: {
      Tensor grad = mask.new_empty({H_q, L, S});
      at::musa::sum_out(grad, contig_grad_mask_full, {0});
      return grad;
    }
    case MaskType::FLT_NHLS_FULL: {
      return contig_grad_mask_full;
    }
    case MaskType::FLT_HLS_BCAST: {
      const auto mask_sizes = mask.sizes();
      DimVector grad_shape{1};
      grad_shape.insert(grad_shape.end(), mask_sizes.begin(), mask_sizes.end());
      Tensor grad = mask.new_empty(grad_shape);
      DimVector axes;
      if (N != 1) {
        axes.push_back(0);
      }
      const int64_t full_dim = contig_grad_mask_full.dim();
      for (int64_t i = 1; i < full_dim; ++i) {
        if (full_sizes[i] != mask_sizes[i]) {
          axes.push_back(i);
        }
      }
      at::musa::sum_out(grad, contig_grad_mask_full, axes, true);
      return grad.squeeze(0);
    }
    case MaskType::FLT_NHLS_BCAST: {
      const auto mask_sizes = mask.sizes();
      Tensor grad = mask.new_empty(mask_sizes);
      DimVector axes;
      const int64_t full_dim = contig_grad_mask_full.dim();
      for (int64_t i = 0; i < full_dim; ++i) {
        if (full_sizes[i] != mask_sizes[i]) {
          axes.push_back(i);
        }
      }
      at::musa::sum_out(grad, contig_grad_mask_full, axes, true);
      return grad;
    }
    default:
      break;
  }
  return Tensor();
}

void CheckScale(const Tensor& query, std::optional<double> scale) {
#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 4020)
  if (scale.has_value()) {
    using FLT = FloatingPoint<double>;
    using SFT = c10::SymFloat;
    const auto expect_scale = (SFT(1.0) / (SFT(query.sym_size(-1)).sqrt()));
    const auto almost_equal =
        FLT(expect_scale.expect_float()).AlmostEquals(FLT(scale.value()));
    if (!almost_equal) {
      TORCH_CHECK(
          false,
          "Not support explicit `scale` parameter which is ",
          "not equal to default `1/sqrt(query.size(-1)).");
    }
    TORCH_WARN_ONCE(
        "Detected explicit `scale` parameter which is almost equal to default ",
        "`1/sqrt(query.size(-1)), SDPA will treat it as `scale=None`. If this ",
        "is the expected behavior, you can set it manually to avoid this warning.");
  }
#endif
}

void MaybeSetScale(
    ::musa::dnn::ScaledDotProductAttention& sdpa,
    std::optional<double> scale) {
#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION >= 4020)
  if (scale.has_value()) {
    CHECK_MUDNN_STATUS(sdpa.SetScale(scale.value()), "SetScale");
  }
#endif
}

void MathCheckTensorShapes(const Tensor& q, const Tensor& k, const Tensor& v) {
  const auto q_dim = q.dim();
  const auto k_dim = k.dim();
  const auto v_dim = v.dim();
  TORCH_CHECK(
      (q_dim == k_dim) && (q_dim == v_dim) && (q_dim == 3 || q_dim == 4),
      "SDPA on MUSA requires query, key and value to be 3(no batch) ",
      "or 4(with batch) dimensional, but got Query dim: ",
      q_dim,
      ", Key dim: ",
      k_dim,
      ", Value dim: ",
      v_dim,
      " instead.");
}

using Proxy = typename c10::MaybeOwned<Tensor>;
Proxy AffineBatch(const Tensor& orig, bool no_batch) {
  if (no_batch) {
    return Proxy::owned(orig.unsqueeze(0));
  }
  return Proxy::borrowed(orig);
}

} // anonymous namespace

// query shape: [N, H_q, L, E]
// key   shape: [N, H, S, E]
// value shape: [N, H, S, E_v]
std::tuple<Tensor, Tensor, Tensor> MuDNNMathSDPAFwd(
    const Tensor& _query,
    const Tensor& _key,
    const Tensor& _value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  const c10::musa::MUSAGuard device_guard(_query.device());

  MathCheckTensorShapes(_query, _key, _value);
  const auto no_batch = (_query.dim() == 3);
  Proxy batch_q = AffineBatch(_query, no_batch);
  Proxy batch_k = AffineBatch(_key, no_batch);
  Proxy batch_v = AffineBatch(_value, no_batch);
  const auto& query = (*batch_q);
  const auto& key = (*batch_k);
  const auto& value = (*batch_v);

  const auto N = query.size(0);
  const auto H_q = query.size(1);
  const auto L = query.size(2);
  const auto E = query.size(3);

  const auto S = key.size(2);

  const auto E_v = value.size(3);

  auto mask = attn_mask; // remove const
  CheckScale(query, scale);
  const auto query_opt = query.options();
  if (is_causal) {
    TORCH_CHECK(
        !attn_mask.has_value(),
        "MUSA SPDA: Explicit attn_mask should not be set when is_causal=True");
    TORCH_CHECK(
        !query.is_nested() && !key.is_nested(),
        "MUSA SPDA: Nested tensors for query / key are not supported when is_causal=True");
    mask = at::ones({L, S}, query_opt.dtype(at::kBool));
    mask->tril_();
  }
  const auto mask_type = ParseMaskType(mask, is_causal, N, H_q, L, S);

  const auto contig_q = query.contiguous();
  const auto contig_k = key.contiguous();
  const auto contig_v = value.contiguous();

  auto musa_q = CreateMUTensor(contig_q);
  auto musa_k = CreateMUTensor(contig_k);
  auto musa_v = CreateMUTensor(contig_v);

  auto contig_mask =
      mask.has_value() ? mask->contiguous() : at::empty({0}, query_opt);
  if (IsNoBatchMask(mask_type) && no_batch) {
    contig_mask = contig_mask.unsqueeze(0);
  }
  auto musa_mask = CreateMUTensor(contig_mask);

  auto contig_output =
      at::empty({N, H_q, L, E_v}, query_opt, at::MemoryFormat::Contiguous);
  auto musa_out = CreateMUTensor(contig_output);

  auto contig_attn_weights =
      at::empty({N, H_q, L, S}, query_opt, at::MemoryFormat::Contiguous);
  auto musa_attn_weights = CreateMUTensor(contig_attn_weights);

  auto& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  CHECK_MUDNN_STATUS(
      sdpa.SetComputeMode(musa::GetComputeModeFromCtx(query.scalar_type())),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(H_q * E_v), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetMaskMode(IsPadMask(mask_type)), "SetMaskMode");

  MaybeSetScale(sdpa, scale);

  auto contig_dropout_mask = at::empty({0}, query_opt.dtype(at::kBool));
  if (dropout_p > 0.0) {
    contig_dropout_mask.resize_({N, H_q, L, S});
    CHECK_MUDNN_STATUS(sdpa.SetDropoutP(dropout_p), "SetDropoutP");
    CHECK_MUDNN_STATUS(sdpa.SetTraining(true), "SetTraining");
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

  if (no_batch) {
    contig_output.squeeze_(0);
    contig_attn_weights.squeeze_(0);
    contig_dropout_mask.squeeze_(0);
  }
  return std::make_tuple(
      contig_output, contig_attn_weights, contig_dropout_mask);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> MuDNNMathSDPABwd(
    const Tensor& _grad_output,
    const Tensor& _query,
    const Tensor& _key,
    const Tensor& _value,
    const Tensor& _output,
    const Tensor& _attn_weights,
    const Tensor& _dropout_mask,
    bool is_causal,
    const std::optional<Tensor>& attn_mask,
    std::optional<double> scale) {
  const c10::musa::MUSAGuard device_guard(_query.device());

  const auto no_batch = (_query.dim() == 3);
  Proxy batch_q = AffineBatch(_query, no_batch);
  Proxy batch_k = AffineBatch(_key, no_batch);
  Proxy batch_v = AffineBatch(_value, no_batch);
  Proxy batch_go = AffineBatch(_grad_output, no_batch);
  Proxy batch_o = AffineBatch(_output, no_batch);
  Proxy batch_aw = AffineBatch(_attn_weights, no_batch);
  Proxy batch_dm = AffineBatch(_dropout_mask, no_batch);

  const auto& query = (*batch_q);
  const auto& key = (*batch_k);
  const auto& value = (*batch_v);
  const auto& grad_output = (*batch_go);
  const auto& output = (*batch_o);
  const auto& attn_weights = (*batch_aw);
  const auto& dropout_mask = (*batch_dm);

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

  auto contig_grad_attn_weights = at::empty_like(contig_attn_weights);
  auto musa_grad_attn_weights = CreateMUTensor(contig_grad_attn_weights);

  const auto contig_dropout_mask = dropout_mask.contiguous();
  auto musa_dropout_mask = CreateMUTensor(contig_dropout_mask);

  auto& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  const auto N = query.size(0);
  const auto H_q = query.size(1);
  const auto L = query.size(2);

  const auto S = key.size(2);

  const auto E_v = value.size(3);

  const auto mask_type = ParseMaskType(attn_mask, is_causal, N, H_q, L, S);

  CHECK_MUDNN_STATUS(
      sdpa.SetComputeMode(musa::GetComputeModeFromCtx(query.scalar_type())),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(H_q * E_v), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetTraining(true), "SetTraining");
  MaybeSetScale(sdpa, scale);

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

  const bool mask_need_grad =
      IsValidGradMask(mask_type) && attn_mask->requires_grad();
  auto attn_mask_grad = mask_need_grad
      ? CalMaskGrad(contig_grad_attn_weights, *attn_mask, mask_type)
      : Tensor();

  if (no_batch) {
    contig_grad_q.squeeze_(0);
    contig_grad_k.squeeze_(0);
    contig_grad_v.squeeze_(0);
  }

  return std::make_tuple(
      contig_grad_q, contig_grad_k, contig_grad_v, attn_mask_grad);
}

std::tuple<Tensor, Tensor, Tensor> MuDNNFlashSDPAFwd(
    const Tensor& _query,
    const Tensor& _key,
    const Tensor& _value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  const c10::musa::MUSAGuard device_guard(_query.device());

  const auto no_batch = (_query.dim() == 3);
  Proxy batch_q = AffineBatch(_query, no_batch);
  Proxy batch_k = AffineBatch(_key, no_batch);
  Proxy batch_v = AffineBatch(_value, no_batch);
  const auto& query = (*batch_q);
  const auto& key = (*batch_k);
  const auto& value = (*batch_v);

  auto mask = attn_mask; // remove const
  CheckScale(query, scale);
  const auto query_opt = query.options();
  if (is_causal) {
    TORCH_CHECK(
        !attn_mask.has_value(),
        "MUSA SPDA: Explicit attn_mask should not be set when is_causal=True");
    TORCH_CHECK(
        !query.is_nested() && !key.is_nested(),
        "MUSA SPDA: Nested tensors for query / key are not supported when is_causal=True");
  }

  const auto N = query.size(0);
  const auto H_q = query.size(1);
  const auto L = query.size(2);
  const auto E = query.size(3);

  const auto S = key.size(2);

  const auto E_v = value.size(3);

  const auto mask_type = ParseMaskType(mask, is_causal, N, H_q, L, S);

  auto musa_q = CreateMUTensor(query);
  auto musa_k = CreateMUTensor(key);
  auto musa_v = CreateMUTensor(value);

  auto output =
      at::empty({N, L, H_q, E_v}, query_opt, at::MemoryFormat::Contiguous)
          .transpose(1, 2);
  auto musa_out = CreateMUTensor(output);

  auto contig_mask =
      HasMask(mask_type) ? mask->contiguous() : at::empty({0}, query_opt);
  if (IsNoBatchMask(mask_type) && no_batch) {
    contig_mask = contig_mask.unsqueeze(0);
  }
  auto musa_mask = CreateMUTensor(contig_mask);

  auto contig_logsumexp = at::empty(
      {N, H_q, L}, query_opt.dtype(at::kFloat), at::MemoryFormat::Contiguous);
  auto musa_lse = CreateMUTensor(contig_logsumexp);

  auto& h = at::GetMudnnHandle();
  ::musa::dnn::ScaledDotProductAttention sdpa;

  CHECK_MUDNN_STATUS(sdpa.SetCausal(is_causal), "SetCausal");
  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(H_q * E_v), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetMaskMode(IsPadMask(mask_type)), "SetMaskMode");
  MaybeSetScale(sdpa, scale);

  auto contig_dropout_mask = at::empty({0}, query_opt.dtype(at::kBool));
  if (dropout_p > 0.0) {
    contig_dropout_mask.resize_({N, H_q, L, S});
    CHECK_MUDNN_STATUS(sdpa.SetDropoutP(dropout_p), "SetDropoutP");
    CHECK_MUDNN_STATUS(sdpa.SetTraining(true), "SetTraining");
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

  if (no_batch) {
    output.squeeze_(0);
    contig_logsumexp.squeeze_(0);
    contig_dropout_mask.squeeze_(0);
  }
  return std::make_tuple(output, contig_logsumexp, contig_dropout_mask);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> MuDNNFlashSDPABwd(
    const Tensor& _grad_output,
    const Tensor& _query,
    const Tensor& _key,
    const Tensor& _value,
    const Tensor& _output,
    const Tensor& _logsumexp,
    const Tensor& _dropout_mask,
    bool is_causal,
    const std::optional<Tensor>& attn_mask,
    std::optional<double> scale) {
  const c10::musa::MUSAGuard device_guard(_query.device());

  const auto no_batch = (_query.dim() == 3);
  Proxy batch_q = AffineBatch(_query, no_batch);
  Proxy batch_k = AffineBatch(_key, no_batch);
  Proxy batch_v = AffineBatch(_value, no_batch);
  Proxy batch_go = AffineBatch(_grad_output, no_batch);
  Proxy batch_o = AffineBatch(_output, no_batch);
  Proxy batch_lse = AffineBatch(_logsumexp, no_batch);
  Proxy batch_dm = AffineBatch(_dropout_mask, no_batch);

  const auto& query = (*batch_q);
  const auto& key = (*batch_k);
  const auto& value = (*batch_v);
  const auto& grad_output = (*batch_go);
  const auto& output = (*batch_o);
  const auto& logsumexp = (*batch_lse);
  const auto& dropout_mask = (*batch_dm);

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

  const auto N = query.size(0);
  const auto H_q = query.size(1);
  const auto L = query.size(2);

  const auto S = key.size(2);

  const auto E_v = value.size(3);

  const auto mask_type = ParseMaskType(attn_mask, is_causal, N, H_q, L, S);
  auto contig_mask = HasMask(mask_type) ? attn_mask->contiguous()
                                        : at::empty({0}, query.options());
  if (IsNoBatchMask(mask_type) && no_batch) {
    contig_mask = contig_mask.unsqueeze(0);
  }
  auto musa_mask = CreateMUTensor(contig_mask);

  CHECK_MUDNN_STATUS(sdpa.SetEmbedDim(H_q * E_v), "SetEmbedDim");
  CHECK_MUDNN_STATUS(sdpa.SetHeadsNum(H_q), "SetHeadsNum");
  CHECK_MUDNN_STATUS(sdpa.SetTraining(true), "SetTraining");
  CHECK_MUDNN_STATUS(sdpa.SetCausal(is_causal), "SetCausal");
  CHECK_MUDNN_STATUS(sdpa.SetMaskMode(IsPadMask(mask_type)), "SetMaskMode");
  MaybeSetScale(sdpa, scale);

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

  if (no_batch) {
    grad_q.squeeze_(0);
    grad_k.squeeze_(0);
    grad_v.squeeze_(0);
  }

  return std::make_tuple(grad_q, grad_k, grad_v, Tensor());
}

} // namespace at::musa
