#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/core/Tensor.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include "mutlass_ops/gemm/w4a16/w4a16_gemm.hpp"

namespace at::musa {

using namespace mute;

// ---------------------------------------------------------------------------
// dispatch on bias + activation
// ---------------------------------------------------------------------------
template <typename ElementA, typename StrideA, typename StrideC>
static void mixed_dtypes_linear_w4a16_dispatch(
    ElementA const* ptr_a,
    StrideA stride_a,
    std::uint8_t const* ptr_b4,
    float const* ptr_scale,
    ElementA* ptr_c,
    StrideC stride_c,
    ElementA const* ptr_bias,
    int32_t m,
    int32_t n,
    int32_t k,
    int32_t k_packed,
    int32_t scale_k,
    int32_t group_size,
    const std::string_view& activation,
    musaStream_t stream) {
  using namespace mutlass_kernels::collective;

  if (ptr_bias == nullptr && activation == "none") {
    mutlass_kernels::run_w4a16_gemm<ElementA, StrideA, StrideC>(
        ptr_a,
        stride_a,
        ptr_b4,
        ptr_scale,
        ptr_c,
        stride_c,
        m,
        n,
        k,
        k_packed,
        scale_k,
        group_size,
        stream);
    return;
  }

  if (activation == "none") {
    mutlass_kernels::run_w4a16_gemm_bias_act<
        EpilogueIdentity<float>,
        ElementA,
        StrideA,
        StrideC>(
        ptr_a,
        stride_a,
        ptr_b4,
        ptr_scale,
        ptr_c,
        stride_c,
        ptr_bias,
        m,
        n,
        k,
        k_packed,
        scale_k,
        group_size,
        stream);
  } else if (activation == "relu") {
    mutlass_kernels::run_w4a16_gemm_bias_act<
        EpilogueRelu<float>,
        ElementA,
        StrideA,
        StrideC>(
        ptr_a,
        stride_a,
        ptr_b4,
        ptr_scale,
        ptr_c,
        stride_c,
        ptr_bias,
        m,
        n,
        k,
        k_packed,
        scale_k,
        group_size,
        stream);
  } else if (activation == "silu") {
    mutlass_kernels::run_w4a16_gemm_bias_act<
        EpilogueSilu<float>,
        ElementA,
        StrideA,
        StrideC>(
        ptr_a,
        stride_a,
        ptr_b4,
        ptr_scale,
        ptr_c,
        stride_c,
        ptr_bias,
        m,
        n,
        k,
        k_packed,
        scale_k,
        group_size,
        stream);
  } else {
    TORCH_CHECK(
        false,
        "_mixed_dtypes_linear: Activation \"",
        activation,
        "\" is not supported. Supported: none, relu, silu"); // TODO: add gelu
                                                             // epilogue
  }
}

// ---------------------------------------------------------------------------
// Shared validation and kernel launch logic
// ---------------------------------------------------------------------------
// current support:
// 1. int4 weight, fp16/bf16 input, per-group scale
// 2. int4 weight, fp16/bf16 input, per-channel scale
// TODO: int8 weight, fp16/bf16 input
// TODO: int4 weight, fp8 input
// ---------------------------------------------------------------------------
static void mixed_dtypes_linear_impl(
    const at::Tensor& input_2d,
    const at::Tensor& weight,
    const at::Tensor& scale,
    const at::Tensor& bias,
    const std::string_view& activation,
    at::Tensor& output) {
  // ----- Validate layouts -----
  TORCH_CHECK(
      input_2d.layout() == at::Layout::Strided,
      "_mixed_dtypes_linear: Expected input to be strided, but got layout ",
      input_2d.layout());
  TORCH_CHECK(
      input_2d.dim() == 2,
      "_mixed_dtypes_linear: Expected input to be 2D, got ",
      input_2d.dim(),
      " dims");
  const auto strides_input = input_2d.strides();
  TORCH_CHECK(
      strides_input[0] > 1 && strides_input[1] == 1,
      "_mixed_dtypes_linear: Invalid strides for input: row stride = ",
      strides_input[0],
      ", column stride = ",
      strides_input[1]);

  // Weight: prepacked uint8 blob (2D view of tiled layout).
  TORCH_CHECK(
      weight.is_contiguous(),
      "_mixed_dtypes_linear: Expected weight to be contiguous");
  TORCH_CHECK(
      weight.dim() == 2,
      "_mixed_dtypes_linear: Expected pre-packed 2D weight, got ",
      weight.dim(),
      "-D. Use prepack_int4_weight_for_mixed_dtypes_linear().");

  // Scale must be pre-packed: 2D (k_tiles, N), contiguous float32
  TORCH_CHECK(
      scale.dim() == 2,
      "_mixed_dtypes_linear: Expected pre-packed 2D scale, got ",
      scale.dim(),
      "-D. Use prepack_scale_for_mixed_dtypes_linear().");
  if (bias.defined()) {
    TORCH_CHECK(
        bias.dim() == 1,
        "_mixed_dtypes_linear: Expected bias to be 1D, got ",
        bias.dim(),
        " dims");
  }

  // ----- Dimensions -----
  const int32_t length_m = static_cast<int32_t>(input_2d.size(0));
  const int32_t length_k = static_cast<int32_t>(input_2d.size(1));
  const int32_t length_n = static_cast<int32_t>(scale.size(1));

  if (bias.defined()) {
    TORCH_CHECK(
        bias.size(0) == length_n,
        "_mixed_dtypes_linear: Expected bias to have ",
        length_n,
        " elements, but got ",
        bias.size(0));
  }

  // ----- Validate output -----
  TORCH_CHECK(
      output.size(0) == length_m && output.size(1) == length_n,
      "_mixed_dtypes_linear: Expected output shape [",
      length_m,
      ", ",
      length_n,
      "], but got [",
      output.size(0),
      ", ",
      output.size(1),
      "]");
  TORCH_CHECK(
      output.dtype() == input_2d.dtype(),
      "_mixed_dtypes_linear: Expected output dtype ",
      input_2d.dtype(),
      ", but got ",
      output.dtype());
  const auto strides_output = output.strides();
  TORCH_CHECK(
      strides_output[1] == 1,
      "_mixed_dtypes_linear: Expected output to be row-major contiguous");

  const int32_t k_packed = length_k / 2; // int4: 2 elements per byte
  const int32_t scale_k = static_cast<int32_t>(scale.size(0));

  TORCH_CHECK(
      scale.scalar_type() == at::kFloat,
      "_mixed_dtypes_linear: Expected pre-packed float32 scale, got ",
      scale.dtype());
  TORCH_CHECK(
      scale.is_contiguous(),
      "_mixed_dtypes_linear: Expected scale to be contiguous");
  TORCH_CHECK(
      length_k % scale_k == 0,
      "_mixed_dtypes_linear: K (",
      length_k,
      ") must be divisible by scale_k (",
      scale_k,
      ")");
  TORCH_CHECK(
      length_n % 256 == 0,
      "_mixed_dtypes_linear: N (",
      length_n,
      ") must be a multiple of 256 (BlockN of the W4A16 kernel)");
  TORCH_CHECK(
      length_k % 128 == 0,
      "_mixed_dtypes_linear: K (",
      length_k,
      ") must be a multiple of 128 (BlockK of the W4A16 kernel)");
  const int32_t group_size = length_k / scale_k;

  auto stream = at::musa::getCurrentMUSAStream().stream();

  // ----- Stride types -----
  using StrideA = decltype(make_stride(int32_t{}, Int<1>{}));
  using StrideC = decltype(make_stride(int32_t{}, Int<1>{}));
  auto stride_a = make_stride(length_k, Int<1>{});
  auto stride_c = make_stride(length_n, Int<1>{});

  const auto* ptr_b4 = reinterpret_cast<const std::uint8_t*>(weight.data_ptr());
  const auto* ptr_scale = scale.data_ptr<float>();

  // ----- Dispatch on input dtype: just bf16 & fp16 -----
  if (input_2d.scalar_type() == at::ScalarType::Half) {
    using ElementA = mutlass::half_t;
    const auto* ptr_a = reinterpret_cast<const ElementA*>(input_2d.data_ptr());
    auto* ptr_c = reinterpret_cast<ElementA*>(output.data_ptr());
    const auto* ptr_bias = bias.defined()
        ? reinterpret_cast<const ElementA*>(bias.data_ptr())
        : static_cast<const ElementA*>(nullptr);

    mixed_dtypes_linear_w4a16_dispatch<ElementA, StrideA, StrideC>(
        ptr_a,
        stride_a,
        ptr_b4,
        ptr_scale,
        ptr_c,
        stride_c,
        ptr_bias,
        length_m,
        length_n,
        length_k,
        k_packed,
        scale_k,
        group_size,
        activation,
        stream);
  } else {
    using ElementA = mutlass::bfloat16_t;
    const auto* ptr_a = reinterpret_cast<const ElementA*>(input_2d.data_ptr());
    auto* ptr_c = reinterpret_cast<ElementA*>(output.data_ptr());
    const auto* ptr_bias = bias.defined()
        ? reinterpret_cast<const ElementA*>(bias.data_ptr())
        : static_cast<const ElementA*>(nullptr);

    mixed_dtypes_linear_w4a16_dispatch<ElementA, StrideA, StrideC>(
        ptr_a,
        stride_a,
        ptr_b4,
        ptr_scale,
        ptr_c,
        stride_c,
        ptr_bias,
        length_m,
        length_n,
        length_k,
        k_packed,
        scale_k,
        group_size,
        activation,
        stream);
  }

  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

static void validate_dtypes(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& scale,
    const at::Tensor& bias) {
  TORCH_CHECK(
      input.dtype() == at::kHalf || input.dtype() == at::kBFloat16,
      "_mixed_dtypes_linear: The input datatype ",
      input.dtype(),
      " is not supported");
  TORCH_CHECK(
      weight.dtype() == at::kByte,
      "_mixed_dtypes_linear: The weight datatype ",
      weight.dtype(),
      " is not supported");
  TORCH_CHECK(
      scale.dtype() == at::kFloat,
      "_mixed_dtypes_linear: Expected scale datatype Float ",
      ", but got ",
      scale.dtype());
  if (bias.defined()) {
    TORCH_CHECK(
        bias.dtype() == input.dtype(),
        "_mixed_dtypes_linear: Expected bias datatype ",
        input.dtype(),
        " but got ",
        bias.dtype());
  }
}

at::Tensor MixedTypesLinear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& scale,
    const std::optional<at::Tensor>& bias_opt,
    const std::optional<std::string_view> activation_opt) {
  c10::musa::MUSAGuard g(input.device());

  const auto bias = bias_opt.has_value() ? *bias_opt : Tensor{};
  const auto activation = activation_opt.has_value() ? *activation_opt : "none";

  validate_dtypes(input, weight, scale, bias);

  const auto input_sizes = input.sizes().vec();
  const auto input_2d = input.reshape({-1, input_sizes.back()});

  const int32_t length_m = static_cast<int32_t>(input_2d.size(0));
  const int32_t length_n = static_cast<int32_t>(scale.size(1));

  auto output = input_2d.new_empty({length_m, length_n});

  mixed_dtypes_linear_impl(input_2d, weight, scale, bias, activation, output);

  auto output_sizes = input_sizes;
  output_sizes.back() = length_n;
  return output.reshape(output_sizes);
}

} // namespace at::musa
