#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/ops/add.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/div.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/relu.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/util/MaybeOwned.h>
#include <torch/library.h>
#include <vector>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/quantized/mudnn/Linear.h"
#include "torch_musa/csrc/aten/quantized/mudnn/QConv.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace native {
DEFINE_DISPATCH(conv_stub);
REGISTER_NO_CPU_DISPATCH(conv_stub);

DEFINE_DISPATCH(conv_binary_stub);
REGISTER_NO_CPU_DISPATCH(conv_binary_stub);
} // namespace native

namespace musa {
static at::Tensor run_pointwise(
    at::Tensor act,
    double act_scale,
    int64_t act_zero_point,
    at::Tensor weight,
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    std::optional<at::Tensor> bias,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::string_view attr,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<std::string_view> algorithm) {
  TORCH_CHECK(
      attr == "none" || attr == "relu",
      "We support quantized convolution without any post-ops or combinations for Quantized Conv + ReLU. However, encountered unsupported post operation:",
      attr,
      ".");
  const int64_t dim = act.dim();
  TORCH_CHECK(dim == 4, "qconv musa: input dim should be 4, but got", dim);

  std::vector<int64_t> stride_vec(stride.begin(), stride.end());
  std::vector<int64_t> padding_vec(padding.begin(), padding.end());
  std::vector<int64_t> dilation_vec(dilation.begin(), dilation.end());
  c10::IntArrayRef stride_symint(stride_vec);
  c10::IntArrayRef padding_symint(padding_vec);
  c10::IntArrayRef dilation_symint(dilation_vec);
  auto dst_dtype = output_dtype.value_or(act.scalar_type());
  if (act.scalar_type() == at::kChar || weight.scalar_type() == at::kByte) {
    // kernel now only support act uint8 weight int8, if needed, can add later
    at::Tensor dequant_input =
        (act.to(c10::kFloat) - act_zero_point) * act_scale;
    at::Tensor dequant_weight =
        (weight.to(c10::kFloat) - weight_zero_points.view({-1, 1, 1, 1})) *
        weight_scales.view({-1, 1, 1, 1});
    at::Tensor conv = at::conv2d(
        dequant_input,
        dequant_weight,
        bias,
        stride_symint,
        padding_symint,
        dilation_symint,
        groups);
    if (attr == "relu") {
      conv = at::relu(conv);
    }
    int64_t clamp_min, clamp_max;
    switch (dst_dtype) {
      case c10::ScalarType::Byte:
        clamp_min = 0;
        clamp_max = 255;
        break;
      case c10::ScalarType::Char:
        clamp_min = -128;
        clamp_max = 127;
        break;
      default:
        conv = conv.to(dst_dtype);
        return conv;
    }
    at::Tensor quantized_output = (conv / output_scale + output_zero_point)
                                      .round()
                                      .clamp(clamp_min, clamp_max)
                                      .to(dst_dtype);
    return quantized_output;
  } else {
    int64_t batch_size = act.size(0);
    int64_t in_h = act.size(2);
    int64_t in_w = act.size(3);
    // [batch_size, channel, height, width]
    int64_t kernel_h = weight.size(2);
    int64_t kernel_w = weight.size(3);

    int64_t out_c = weight.size(0);

    Tensor b_raw = bias.has_value() ? bias.value() : at::Tensor();

    int64_t out_h =
        (in_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) / stride[0] +
        1;
    int64_t out_w =
        (in_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) / stride[1] +
        1;

    std::vector<int64_t> dst_dims = {batch_size, out_c, out_h, out_w};
    at::Tensor output = at::empty(dst_dims, act.options().dtype(dst_dtype));

    at::native::conv_stub(
        at::kMUSA,
        act,
        act_scale,
        act_zero_point,
        weight,
        weight_scales,
        weight_zero_points,
        b_raw,
        output,
        stride_symint,
        padding_symint,
        dilation_symint,
        groups,
        output_scale,
        output_zero_point,
        dst_dtype,
        attr);
    return output;
  }
}

static at::Tensor run_pointwise_binary(
    at::Tensor act,
    double act_scale,
    int64_t act_zero_point,
    at::Tensor weight,
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    at::Tensor accum,
    std::optional<at::Tensor> bias,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    double accum_scale,
    int64_t accum_zero_point,
    std::string_view binary_attr,
    std::optional<at::Scalar> alpha,
    std::optional<std::string_view> unary_attr,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<std::string_view> algorithm) {
  TORCH_CHECK(
      act.dim() == 4 && binary_attr == "sum" &&
          (!unary_attr.has_value() ||
           (unary_attr.has_value() &&
            (unary_attr.value() == "none" || unary_attr.value() == "relu"))),
      "post_op sum or post_op sum_relu is supported for quantized pointwise conv2d. Got binary_post_op: ",
      binary_attr,
      " unary_post_op: ",
      unary_attr.has_value() ? unary_attr.value() : "none",
      ".")
  std::vector<int64_t> stride_vec(stride.begin(), stride.end());
  std::vector<int64_t> padding_vec(padding.begin(), padding.end());
  std::vector<int64_t> dilation_vec(dilation.begin(), dilation.end());
  c10::IntArrayRef stride_symint(stride_vec);
  c10::IntArrayRef padding_symint(padding_vec);
  c10::IntArrayRef dilation_symint(dilation_vec);
  double alpha_value = alpha.has_value() ? alpha.value().to<double>() : 1.0;
  std::string_view unary_op =
      unary_attr.has_value() ? unary_attr.value() : "none";

  if (act.scalar_type() == at::kChar || weight.scalar_type() == at::kByte) {
    at::Tensor dequant_input =
        (act.to(c10::kFloat) - act_zero_point) * act_scale;
    at::Tensor dequant_weight =
        (weight.to(c10::kFloat) - weight_zero_points.view({-1, 1, 1, 1})) *
        weight_scales.view({-1, 1, 1, 1});
    auto dst_dtype = output_dtype.value_or(act.scalar_type());
    TORCH_CHECK(
        (dst_dtype == c10::kFloat || dst_dtype == c10::kBFloat16)
            ? accum_scale == 1.0
            : true,
        "fp32 or bf16 output, accum_scale must be 1.0.")
    accum = (accum.to(c10::kFloat) - accum_zero_point) * accum_scale;
    at::Tensor conv = at::conv2d(
        dequant_input,
        dequant_weight,
        bias,
        stride_symint,
        padding_symint,
        dilation_symint,
        groups);
    if (binary_attr == "sum") {
      conv = conv + alpha_value * accum;
    }
    if (unary_op == "relu") {
      conv = at::relu(conv);
    }
    int64_t clamp_min, clamp_max;
    switch (dst_dtype) {
      case c10::ScalarType::Byte:
        clamp_min = 0;
        clamp_max = 255;
        break;
      case c10::ScalarType::Char:
        clamp_min = -128;
        clamp_max = 127;
        break;
      default:
        conv = conv.to(dst_dtype);
        return conv;
    }
    conv = (conv / output_scale + output_zero_point)
               .round()
               .clamp(clamp_min, clamp_max)
               .to(dst_dtype);
    return conv;
  } else {
    int64_t batch_size = act.size(0);
    int64_t in_h = act.size(2);
    int64_t in_w = act.size(3);
    // [batch_size, channel, height, width]
    int64_t kernel_h = weight.size(2);
    int64_t kernel_w = weight.size(3);

    int64_t out_c = weight.size(0);

    Tensor b_raw = bias.has_value() ? bias.value() : at::Tensor();

    int64_t out_h =
        (in_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) / stride[0] +
        1;
    int64_t out_w =
        (in_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) / stride[1] +
        1;

    std::vector<int64_t> dst_dims = {batch_size, out_c, out_h, out_w};
    auto dst_dtype = output_dtype.value_or(act.scalar_type());
    at::Tensor output = at::empty(dst_dims, act.options().dtype(dst_dtype));

    at::native::conv_binary_stub(
        at::kMUSA,
        act,
        act_scale,
        act_zero_point,
        weight,
        weight_scales,
        weight_zero_points,
        accum,
        b_raw,
        output,
        stride_symint,
        padding_symint,
        dilation_symint,
        groups,
        output_scale,
        output_zero_point,
        dst_dtype,
        binary_attr,
        accum_scale,
        accum_zero_point,
        alpha_value,
        unary_op);
    return output;
  }
}

TORCH_LIBRARY_IMPL(quantized, PrivateUse1, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv"), TORCH_FN(run_pointwise));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv_binary"),
      TORCH_FN(run_pointwise_binary));
}
} // namespace musa
} // namespace at