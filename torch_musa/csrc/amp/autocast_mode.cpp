#include "torch_musa/csrc/amp/autocast_mode.h"

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>

#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/intrusive_ptr.h>

#include <exception>
#include <mutex>

namespace at::autocast {
namespace musa {

// autocast_cpu_dtype is the lower_precision_fp used by AutocastCPU.
thread_local at::ScalarType autocast_musa_dtype = at::kBFloat16;

bool is_autocast_musa_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(
      DispatchKey::AutocastPrivateUse1);
}

void set_autocast_musa_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(
      DispatchKey::AutocastPrivateUse1, !new_enabled);
}

at::ScalarType get_autocast_musa_dtype() {
  return at::autocast::get_autocast_dtype(at::kPrivateUse1);
}

void set_autocast_musa_dtype(at::ScalarType dtype) {
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat,
      "Currently, AutoCastMusa only support float16/bfloat16/float32 as the autocast_musa_dtype");
  return at::autocast::set_autocast_dtype(at::kPrivateUse1, dtype);
}

/*******************************
Banned functions
*******************************/

Tensor binary_cross_entropy_banned(
    const Tensor&,
    const Tensor&,
    const c10::optional<Tensor>&,
    int64_t) {
  AT_ERROR(
      "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
      "Many models use a sigmoid layer right before the binary cross entropy layer.\n"
      "In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits\n"
      "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\n"
      "safe to autocast.");
}

} // namespace musa

namespace {

/*****************************************
Explicit registration for out-of-place ops
*****************************************/
TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
  // lower_precision_fp
  KERNEL_MUSA(_convolution, lower_precision_fp)
  KERNEL_MUSA(conv1d, lower_precision_fp)
  KERNEL_MUSA(conv2d, lower_precision_fp)
  KERNEL_MUSA(conv3d, lower_precision_fp)
  KERNEL_MUSA(conv_tbc, lower_precision_fp)
  KERNEL_MUSA(conv_transpose1d, lower_precision_fp)
  KERNEL_MUSA_FOR_MULTIFORM(conv_transpose2d, input, lower_precision_fp)
  KERNEL_MUSA_FOR_MULTIFORM(conv_transpose3d, input, lower_precision_fp)
  KERNEL_MUSA(convolution, lower_precision_fp)
  KERNEL_MUSA(prelu, lower_precision_fp)
  KERNEL_MUSA(addmm, lower_precision_fp)
  KERNEL_MUSA(addmv, lower_precision_fp)
  KERNEL_MUSA(addr, lower_precision_fp)
  KERNEL_MUSA(matmul, lower_precision_fp)
  KERNEL_MUSA(einsum, lower_precision_fp)
  KERNEL_MUSA(mm, lower_precision_fp)
  KERNEL_MUSA(mv, lower_precision_fp)
  KERNEL_MUSA(linear, lower_precision_fp)
  KERNEL_MUSA(addbmm, lower_precision_fp)
  KERNEL_MUSA(baddbmm, lower_precision_fp)
  KERNEL_MUSA(bmm, lower_precision_fp)
  KERNEL_MUSA(chain_matmul, lower_precision_fp)
  KERNEL_MUSA(linalg_multi_dot, lower_precision_fp)
  KERNEL_MUSA(_thnn_fused_lstm_cell, lower_precision_fp)
  KERNEL_MUSA(_thnn_fused_gru_cell, lower_precision_fp)
  KERNEL_MUSA(lstm_cell, lower_precision_fp)
  KERNEL_MUSA(gru_cell, lower_precision_fp)
  KERNEL_MUSA(rnn_tanh_cell, lower_precision_fp)
  KERNEL_MUSA(rnn_relu_cell, lower_precision_fp)
  KERNEL_MUSA(_scaled_dot_product_flash_attention, lower_precision_fp)
  KERNEL_MUSA(scaled_dot_product_attention, lower_precision_fp)

  // fp32
  KERNEL_MUSA(acos, fp32)
  KERNEL_MUSA(asin, fp32)
  KERNEL_MUSA(cosh, fp32)
  KERNEL_MUSA(erfinv, fp32)
  KERNEL_MUSA(exp, fp32)
  KERNEL_MUSA(expm1, fp32)
  KERNEL_MUSA(log, fp32)
  KERNEL_MUSA(log10, fp32)
  KERNEL_MUSA(log2, fp32)
  KERNEL_MUSA(log1p, fp32)
  KERNEL_MUSA(reciprocal, fp32)
  KERNEL_MUSA(rsqrt, fp32)
  KERNEL_MUSA(sinh, fp32)
  KERNEL_MUSA(tan, fp32)
  KERNEL_MUSA_FOR_MULTIFORM(pow, Tensor_Scalar, fp32)
  KERNEL_MUSA_FOR_MULTIFORM(pow, Tensor_Tensor, fp32)
  KERNEL_MUSA_FOR_MULTIFORM(pow, Scalar, fp32)
  KERNEL_MUSA(softplus, fp32)
  KERNEL_MUSA(layer_norm, fp32)
  KERNEL_MUSA(native_layer_norm, fp32)
  KERNEL_MUSA(group_norm, fp32)
  KERNEL_MUSA_FOR_MULTIFORM(frobenius_norm, dim, fp32)
  KERNEL_MUSA(nuclear_norm, fp32)
  KERNEL_MUSA_FOR_MULTIFORM(nuclear_norm, dim, fp32)
  KERNEL_MUSA(cosine_similarity, fp32)
  KERNEL_MUSA(poisson_nll_loss, fp32)
  KERNEL_MUSA(cosine_embedding_loss, fp32)
  KERNEL_MUSA(nll_loss, fp32)
  KERNEL_MUSA(nll_loss2d, fp32)
  KERNEL_MUSA(hinge_embedding_loss, fp32)
  KERNEL_MUSA(kl_div, fp32)
  KERNEL_MUSA(l1_loss, fp32)
  KERNEL_MUSA(smooth_l1_loss, fp32)
  KERNEL_MUSA(huber_loss, fp32)
  KERNEL_MUSA(mse_loss, fp32)
  KERNEL_MUSA(margin_ranking_loss, fp32)
  KERNEL_MUSA(multilabel_margin_loss, fp32)
  KERNEL_MUSA(soft_margin_loss, fp32)
  KERNEL_MUSA(triplet_margin_loss, fp32)
  KERNEL_MUSA(multi_margin_loss, fp32)
  KERNEL_MUSA(binary_cross_entropy_with_logits, fp32)
  KERNEL_MUSA(dist, fp32)
  KERNEL_MUSA(pdist, fp32)
  KERNEL_MUSA(cdist, fp32)
  KERNEL_MUSA(renorm, fp32)
  // fp32_set_opt_dtype
  KERNEL_MUSA(prod, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(prod, dim_int, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(prod, dim_Dimname, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(softmax, int, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(softmax, Dimname, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(log_softmax, int, fp32_set_opt_dtype)

  KERNEL_MUSA_FOR_MULTIFORM(log_softmax, Dimname, fp32_set_opt_dtype)
  KERNEL_MUSA(cumprod, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(cumprod, dimname, fp32_set_opt_dtype)
  KERNEL_MUSA(cumsum, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(cumsum, dimname, fp32_set_opt_dtype)
  KERNEL_MUSA(linalg_vector_norm, fp32_set_opt_dtype)
  KERNEL_MUSA(linalg_matrix_norm, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(linalg_matrix_norm, str_ord, fp32_set_opt_dtype)

  // TODO(kangchen): sum calcuate error when the output and self dtype
  // are different It will get inf, which cuda registers fp32_set_opt_dtype,
  // So I registered as fp32 temporarily.
  KERNEL_MUSA(sum, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(sum, dim_IntList, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(sum, dim_DimnameList, fp32_set_opt_dtype)

  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.

  // norm does not implicitly promote, but be aware when adding new ops to this
  // policy.
  KERNEL_MUSA_DIFFERENT_REDISPATCH_SIGNATURE(
      ADD_NS(norm),
      "norm.Scalar",
      Tensor(const Tensor&, const Scalar&),
      Tensor(const Tensor&, const c10::optional<Scalar>&, ScalarType),
      fp32_append_dtype)
  KERNEL_MUSA_DIFFERENT_REDISPATCH_SIGNATURE(
      ADD_NS(norm),
      "norm.ScalarOpt_dim",
      Tensor(const Tensor&, const c10::optional<Scalar>&, IntArrayRef, bool),
      Tensor(
          const Tensor&,
          const c10::optional<Scalar>&,
          IntArrayRef,
          bool,
          ScalarType),
      fp32_append_dtype)
  KERNEL_MUSA_DIFFERENT_REDISPATCH_SIGNATURE(
      ADD_NS(norm),
      "norm.names_ScalarOpt_dim",
      Tensor(const Tensor&, const c10::optional<Scalar>&, DimnameList, bool),
      Tensor(
          const Tensor&,
          const c10::optional<Scalar>&,
          DimnameList,
          bool,
          ScalarType),
      fp32_append_dtype)
  // promote
  KERNEL_MUSA(addcdiv, promote)
  KERNEL_MUSA(addcmul, promote)
  KERNEL_MUSA(atan2, promote)
  KERNEL_MUSA(bilinear, promote)
  KERNEL_MUSA(cross, promote)
  KERNEL_MUSA(dot, promote)
  KERNEL_MUSA(grid_sampler, promote)
  KERNEL_MUSA(index_put, promote)
  KERNEL_MUSA(tensordot, promote)
  KERNEL_MUSA(scatter_add, promote)

  m.impl(
      TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
      TORCH_FN((&at::autocast::musa::binary_cross_entropy_banned)));
}
} // namespace
} // namespace at::autocast
