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

#include "torch_musa/csrc/aten/musa/MUSAContext.h"

namespace sdp {

inline bool check_musa_arch(const sdp_params& params, bool is_debug) {
  // Check that the gpu is capable of running flash attention
  const auto device_prop = at::musa::getCurrentDeviceProperties();
  const auto arch_major = device_prop->major;
  const auto arch_minor = device_prop->minor;
  const bool enable_flash_attn =
      (arch_major > 2) || (arch_major == 2 && arch_minor > 1);

  if (!enable_flash_attn) {
    if (is_debug) {
      TORCH_WARN(
          "Flash attention only supports architecture with mp version >= 2.2, "
          "but now attempts to run on a mp ",
          arch_major,
          ".",
          arch_minor,
          " gpu.");
    }
    return false;
  }
  return true;
}

inline bool check_musa_attention_input(
    const sdp_params& params,
    bool is_debug) {
  int64_t qk_head_dim = params.query.size(-1);
  int64_t v_head_dim = params.value.size(-1);

  bool is_head_dim_qk192_v128 = qk_head_dim == 192 && v_head_dim == 128;
  bool is_head_dim_qkv_160 = qk_head_dim == 160 && v_head_dim == 160;

  if (!((qk_head_dim <= 128 && v_head_dim <= 128) ||
        (at::musa::getMUSAArch() >= 220 &&
         (is_head_dim_qk192_v128 || is_head_dim_qkv_160)))) {
    if (is_debug) {
      TORCH_WARN(
          "Unsupported qk_head_dim: ",
          qk_head_dim,
          " v_head_dim: ",
          v_head_dim,
          " for FlashAttention in MUSA backend");
    }
    return false;
  }
  return true;
}

inline bool use_flash_attention(const sdp_params& params) {
  using SDPParamsCheckFunc = bool (*)(const sdp_params&, bool);
  constexpr int conditions_num = 4;
  constexpr std::array<SDPParamsCheckFunc, conditions_num> conditions{
      {check_runtime_disabled_flash,
       check_tensor_shapes,
       check_musa_attention_input,
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
  return check_tensor_dtype(params, musa_allowed_dtypes, true);
}

inline SDPBackend select_backend(const sdp_params& params) {
  const auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP() &&
      !ctx.userEnabledMemEfficientSDP()) {
    return SDPBackend::error;
  }
  if (use_flash_attention(params)) {
    return SDPBackend::flash_attention;
  } else if (ctx.userEnabledMathSDP()) {
    return SDPBackend::math;
  } else {
    TORCH_CHECK(false, "Invalid backend configuration for musa!")
  }
}

} // namespace sdp

#endif // TORCH_MUSA_CSRC_ATEN_OPS_ATTENTION_MUDNN_SDPUTILS_H_
