#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/quantized/QTensor.h"
#include "torch_musa/csrc/aten/quantized/mudnn/Conv.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

template <int kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> PackedConvWeightMudnn<
    kSpatialDim>::
    prepack(
        at::Tensor weight,
        c10::optional<at::Tensor> bias,
        torch::List<int64_t> stride,
        torch::List<int64_t> padding,
        torch::List<int64_t> output_padding,
        torch::List<int64_t> dilation,
        int64_t groups,
        bool transpose) {
  TORCH_CHECK(
      weight.qscheme() == c10::kPerTensorAffine ||
          weight.qscheme() == c10::kPerTensorSymmetric,
      "Unsupported qscheme: ",
      toString(weight.qscheme()));
  TORCH_CHECK(
      kSpatialDim == 2, // 1D is packed as 2d, hence we don't need other checks
      "muDNN packing only supports 2D convolution.");
  TORCH_CHECK(
      weight.ndimension() == kSpatialDim + 2,
      "Weights are expected to have ",
      kSpatialDim + 2,
      " dimensions");
  TORCH_CHECK(
      stride.size() == kSpatialDim,
      "stride should contain ",
      kSpatialDim,
      " elements for ",
      kSpatialDim,
      "D convolution.");
  TORCH_CHECK(
      padding.size() == kSpatialDim,
      "quantized::conv_prepack (mudnn): Specify front/top/left padding only. "
      "end/bottom/right padding assumed to be equal to front/top/left");
  TORCH_CHECK(
      !transpose || output_padding.size() == kSpatialDim,
      "quantized::conv_prepack: Specify top/left output padding "
      "only. bottom/right padding assumed to be equal to top/left");
  TORCH_CHECK(
      dilation.size() == kSpatialDim,
      "quantized::conv_prepack (mudnn): dilation should contain ",
      kSpatialDim,
      " elements for ",
      kSpatialDim,
      "D convolution.");
  TORCH_CHECK(
      !transpose, "mudnn quantized conv prepack expects transpose = false")

  TORCH_CHECK(
      weight.scalar_type() == c10::kQInt8,
      "TORCH_MUSA_ARCH > 210 requires weights in format QInt8, which is ",
      weight.scalar_type());
  TORCH_CHECK(
      weight.q_zero_point() == 0,
      "TORCH_MUSA_ARCH > 210 requires weights are quantized in per-tensor and symmetric quantization, ",
      "zero_point should be 0, which is ",
      weight.q_zero_point());

  const int output_channels = weight.size(0);
  const auto qtype = weight.qscheme();
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias.value().size(0) == output_channels,
        "bias should have K elements: " + std::to_string(output_channels));
  }
  weight = weight.permute({0, 2, 3, 1}).contiguous();

  auto ret_ptr = c10::make_intrusive<PackedConvWeightMudnn<kSpatialDim>>(
      weight,
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      transpose,
      qtype,
      output_channels);
  return ret_ptr;
}

template c10::intrusive_ptr<ConvPackedParamsBase<2>> PackedConvWeightMudnn<2>::
    prepack(
        at::Tensor weight,
        c10::optional<at::Tensor> bias_in,
        torch::List<int64_t> stride,
        torch::List<int64_t> padding,
        torch::List<int64_t> output_padding,
        torch::List<int64_t> dilation,
        int64_t groups,
        bool transpose);

namespace at {
namespace musa {
namespace {

template <int kSpatialDim = 2>
class QConvPackWeightInt8Mudnn final {
 public:
  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> run_conv(
      Tensor weight,
      c10::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    c10::musa::MUSAGuard device_guard(weight.device());
    torch::List<int64_t> output_padding;
    output_padding.reserve(kSpatialDim);
    for (const auto idx : c10::irange(kSpatialDim)) {
      (void)idx; // Suppress unused variable warning
      output_padding.push_back((int64_t)0);
    }
    return run(
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        /*transpose=*/false);
  }

 private:
  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> run(
      Tensor weight,
      c10::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose) {
    return PackedConvWeightMudnn<kSpatialDim>::prepack(
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        transpose);
  }
};

class QConv1dPackWeightInt8Mudnn final {
 public:
  static c10::intrusive_ptr<ConvPackedParamsBase<2>> run_conv(
      Tensor weight,
      c10::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    c10::musa::MUSAGuard device_guard(weight.device());
    if (bias.has_value()) {
      bias.value().device();
    }
    const torch::List<int64_t> output_padding({0});
    return run(
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        /*transpose=*/false);
  }

 private:
  static c10::intrusive_ptr<ConvPackedParamsBase<2>> run(
      Tensor weight,
      c10::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose) {
    if (weight.dim() == 3) {
      // we currently use conv2d kernel for conv1d by making the input and
      // weight tensors 4D rather than 3D. we add a dummy width dimension of
      // size 1 out channels, in channels / groups, L -> out channels, in
      // channels / groups, 1, L
      weight = weight.unsqueeze(-2);
    }
    stride = quant_utils::MakeArgForConv1d(stride, 1);
    padding = quant_utils::MakeArgForConv1d(padding, 0);
    output_padding = quant_utils::MakeArgForConv1d(output_padding, 0);
    dilation = quant_utils::MakeArgForConv1d(dilation, 1);

    return PackedConvWeightMudnn<2>::prepack(
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        transpose);
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv1d_prepack"),
      TORCH_FN(QConv1dPackWeightInt8Mudnn::run_conv));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv2d_prepack"),
      TORCH_FN(QConvPackWeightInt8Mudnn<2>::run_conv));
}

} // namespace
} // namespace musa
} // namespace at
