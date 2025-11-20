#include <ATen/autocast_mode.h>

namespace at::autocast {
namespace musa {

Tensor binary_cross_entropy_banned(
    const Tensor&,
    const Tensor&,
    const c10::optional<Tensor>&,
    int64_t) {
  TORCH_CHECK(
      false,
      "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
      "Many models use a sigmoid layer right before the binary cross entropy layer.\n"
      "In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits\n"
      "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\n"
      "safe to autocast.");
}

} // namespace musa

namespace {

TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

#define KERNEL_MUSA(...) KERNEL(c10::DeviceType::PrivateUse1, __VA_ARGS__)

#define KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_MUSA( \
    REDISPATCH_FUNC,                                \
    REGISTER_NAME,                                  \
    REGISTER_SIGNATURE,                             \
    REDISPATCH_SIGNATURE,                           \
    POLICY)                                         \
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(            \
      c10::DeviceType::PrivateUse1,                 \
      REDISPATCH_FUNC,                              \
      REGISTER_NAME,                                \
      REGISTER_SIGNATURE,                           \
      REDISPATCH_SIGNATURE,                         \
      POLICY)

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
  // lower_precision_fp
#define _KERNEL_MUSA_LOW_PRECISION_FP(...) \
  KERNEL_MUSA(__VA_ARGS__, lower_precision_fp)

  AT_FORALL_LOWER_PRECISION_FP(_KERNEL_MUSA_LOW_PRECISION_FP)
  KERNEL_MUSA(convolution_overrideable, lower_precision_fp)
  KERNEL_MUSA(_scaled_dot_product_attention_math_musa, lower_precision_fp)
  KERNEL_MUSA(_scaled_dot_product_attention_flash_musa, lower_precision_fp)

  // fp32
#define _KERNEL_MUSA_FP32(...) KERNEL_MUSA(__VA_ARGS__, fp32)

  AT_FORALL_FP32(_KERNEL_MUSA_FP32)

  // fp32_set_opt_dtype
#define _KERNEL_MUSA_FP32_SET_OPT_DTYPE(...) \
  KERNEL_MUSA(__VA_ARGS__, fp32_set_opt_dtype)

  AT_FORALL_FP32_SET_OPT_DTYPE(_KERNEL_MUSA_FP32_SET_OPT_DTYPE)

  AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(
      KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_MUSA)

  // promote
#define _KERNEL_MUSA_PROMOTE(...) KERNEL_MUSA(__VA_ARGS__, promote)

  AT_FORALL_PROMOTE(_KERNEL_MUSA_PROMOTE)

  m.impl(
      TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
      TORCH_FN((&at::autocast::musa::binary_cross_entropy_banned)));
}

#undef _KERNEL_MUSA_PROMOTE
#undef _KERNEL_MUSA_FP32_SET_OPT_DTYPE
#undef _KERNEL_MUSA_FP32
#undef _KERNEL_MUSA_LOW_PRECISION_FP
#undef KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_MUSA
#undef KERNEL_MUSA

} // namespace
} // namespace at::autocast
