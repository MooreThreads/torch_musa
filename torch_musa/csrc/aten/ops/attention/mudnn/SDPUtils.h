#ifndef TORCH_MUSA_CSRC_ATEN_OPS_ATTENTION_MUDNN_SDPUTILS_H_
#define TORCH_MUSA_CSRC_ATEN_OPS_ATTENTION_MUDNN_SDPUTILS_H_

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_sdp_choice_native.h>
#endif
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/musa/sdp_utils.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

namespace sdp {

inline bool check_musa_arch(sdp_params params, bool is_debug) {
  // Check that the gpu is capable of running flash attention
  auto dprops = at::musa::getCurrentDeviceProperties();
  bool is_arch_greater_than_qy2 =
      dprops->major == 2 && dprops->minor >= 2; // QY2:  major:2 minor:2
  if (!is_arch_greater_than_qy2) {
    if (is_debug) {
      TORCH_WARN(
          "Flash attention only supports QY2 architecture with mp version 2.2. "
          "But now attempts to run on a mp ",
          dprops->major,
          ".",
          dprops->minor,
          " gpu.");
    }
    return false;
  }
  return true;
}

inline bool check_musa_attention_input(sdp::sdp_params params, bool is_debug) {
  auto head_dim = params.query.size(-1);
  if (!(head_dim == 160 || head_dim <= 128)) {
    if (is_debug) {
      TORCH_WARN(
          "FlashAttention in MUSA backend now requires head_dim to be less equal to 128, but got: ",
          head_dim);
    }
    return false;
  }
  return true;
}

inline bool use_flash_attention(const sdp::sdp_params& params) {
  using SDPParamsCheckFunc = bool (*)(sdp::sdp_params, bool);
  constexpr int conditions_num = 4;
  constexpr std::array<SDPParamsCheckFunc, conditions_num> conditions{
      {sdp::check_runtime_disabled_flash,
       sdp::check_tensor_shapes,
       sdp::check_musa_attention_input,
       check_musa_arch}};
  auto res = std::all_of(
      conditions.begin(), conditions.end(), [&params](SDPParamsCheckFunc func) {
        return func(params, true);
      });
  if (!res) {
    return false;
  }
  static const std::array<at::ScalarType, 2> musa_allowed_dtypes{
      at::kHalf, at::kBFloat16};
  return sdp::check_tensor_dtype(params, musa_allowed_dtypes, true);
}

inline sdp::SDPBackend select_backend(const sdp::sdp_params& params) {
  const auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP() &&
      !ctx.userEnabledMemEfficientSDP()) {
    return sdp::SDPBackend::error;
  }
  if (use_flash_attention(params)) {
    return sdp::SDPBackend::flash_attention;
  } else if (ctx.userEnabledMathSDP()) {
    return sdp::SDPBackend::math;
  } else {
    TORCH_CHECK(false, "Invalid backend configuration for musa!")
  }
}
} // namespace sdp

#endif // TORCH_MUSA_CSRC_ATEN_OPS_ATTENTION_MUDNN_SDPUTILS_H_
