#include <ATen/musa/Exceptions.h>
#include <ATen/native/transformers/attention.h>

#include "torch_musa/csrc/aten/ops/attention/mudnn/SDPUtils.h"

namespace at {
namespace native {

int64_t _fused_sdp_choice_musa(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal) {
  sdp::sdp_params params{
      query, key, value, attn_mask.has_value(), dropout_p, is_causal};
  auto backend = sdp::select_backend(params);
  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false,
        "No viable backend for scaled_dot_product_attention in musa",
        "This is likely due to turning off both the math kernel and the flash kernels.");
  }
  return static_cast<int64_t>(backend);
}

REGISTER_MUSA_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_musa);

} // namespace native
} // namespace at