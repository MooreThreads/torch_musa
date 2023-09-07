#include <c10/util/ArrayRef.h>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <c10/util/SmallVector.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/quantized/mudnn/Conv.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <array>
#include <vector>

#include <mudnn.h>

template <>
at::SmallVector<int64_t, 4> MakeQConvOutputShape<2>(
    int N, // mini-batch
    int M, // output channels
    const std::array<int64_t, 2>& input_image_shape,
    const std::vector<int64_t>& kernel,
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& padding,
    const torch::List<int64_t>& dilation) {
  const int H = input_image_shape[0];
  const int W = input_image_shape[1];
  const int64_t Y_H =
      (H + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1;
  const int64_t Y_W =
      (W + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1;
  return {N, M, Y_H, Y_W};
}

template <int kSpatialDim>
template <ActMode act_mode>
void PackedConvWeightMudnn<kSpatialDim>::apply_impl_helper(
    at::Tensor& quantized_output,
    const at::Tensor& input,
    const c10::optional<at::Tensor>& accum,
    double output_scale,
    int64_t output_zero_point) {
  // scale(s) and zero_point(s) of activation and weight
  TORCH_CHECK(
      input.qscheme() == c10::kPerTensorAffine ||
          input.qscheme() == c10::kPerTensorSymmetric,
      "Conv2d only supports per-tensor quantized activation");

  auto act_scale = input.q_scale();
  auto act_zero_point = input.q_zero_point();
  auto weight_scale = weight_.q_scale();
  auto weight_zero_point = weight_.q_zero_point();

  // permute to NHWC format
  at::Tensor input_ = input.permute({0, 2, 3, 1});
  at::musa::muTensor in = at::musa::CreateMUTensor(input_);
  at::musa::muTensor ke = at::musa::CreateMUTensor(weight_);
  at::musa::muTensor out = at::musa::CreateMUTensor(quantized_output);
  // set quantization info (scale and zero_point) to muTensors
  // and set their format
  CHECK_MUDNN_STATUS(
      in.SetFormat(at::musa::muTensor::Format::NHWC),
      "Set input muTensor format as NHWC");
  CHECK_MUDNN_STATUS(
      out.SetFormat(at::musa::muTensor::Format::NHWC),
      "Set output muTensor format as NHWC");
  CHECK_MUDNN_STATUS(
      ke.SetFormat(at::musa::muTensor::Format::NHWC),
      "Set weight muTensor format as NHWC");
  SetMudnnQuantizationInfo(in, act_scale, act_zero_point);
  SetMudnnQuantizationInfo(out, output_scale, output_zero_point);
  SetMudnnQuantizationInfo(ke, weight_scale, weight_zero_point);

  // if bias is used, we should make a muTensor and broadcast it first
  at::musa::muTensor bias;
  if (bias_.has_value() && bias_.value().numel() != 0) {
    bias = at::musa::CreateMUTensor(bias_.value());
  } else {
    bias = at::musa::CreateEmptyMUTensor();
  }

  // mudnn kernel support skip-connection add
  at::musa::muTensor add = at::musa::CreateEmptyMUTensor();
  at::Tensor accum_;
  if (accum.has_value()) {
    TORCH_CHECK(
        accum.value().is_quantized(),
        "accum tensor in qconv must be a quantized Tensor");
    accum_ =
        accum.value().to(c10::MemoryFormat::ChannelsLast).permute({0, 2, 3, 1});
    add = at::musa::CreateMUTensor(accum_);
    CHECK_MUDNN_STATUS(
        add.SetFormat(at::musa::muTensor::Format::NHWC),
        "Set add muTensor format as NHWC");
    SetMudnnQuantizationInfo(
        add, accum.value().q_scale(), accum.value().q_zero_point());
  }

  at::musa::muHandle& h = at::GetMudnnHandle();
  ::musa::dnn::Convolution op;
  ConfigConv(op, padding(), stride(), dilation(), groups());

  // from mudnn, 2 stands for RELU and 0 stands for IDENTITY
  ::musa::dnn::Convolution::FusedActivationDesc act;
  if (ActMode::IDENTITY == act_mode) {
    act.SetMode(
        static_cast<::musa::dnn::Convolution::FusedActivationDesc::Mode>(0));
  } else if (ActMode::RELU == act_mode) {
    act.SetMode(
        static_cast<::musa::dnn::Convolution::FusedActivationDesc::Mode>(2));
  } else if (ActMode::SILU == act_mode) {
    act.SetMode(
        static_cast<::musa::dnn::Convolution::FusedActivationDesc::Mode>(7));
  }

  ::musa::dnn::Convolution::Algorithm algorithm =
      static_cast<::musa::dnn::Convolution::Algorithm>(0);
  size_t size_in_bytes = 0;
  op.GetForwardWorkspaceSize(h, size_in_bytes, out, in, ke, algorithm);
  CHECK_MUDNN_STATUS(
      op.RunFusion(
          h,
          out,
          in,
          ke,
          bias,
          add,
          act,
          algorithm,
          at::musa::InternalMemAlloc),
      "RunFusion");

  return;
}

// weight and output Tensor will be a clampped int8 Tensor
// while QY1 would cast weight to uint8 format
/*
Numerics:
out_fp32  = conv_fp32(act_fp32, w_fp32, …)
          = act_fp32 * w_fp32 + bias_fp32
act_uint8 = act_fp32 / act_scale + act_zero_point
w_uint8    = w_fp32 / w_scale + w_zero_point
out_uint8 = out_fp32 / out_scale + out_zero_point
          = (act_fp32 * w_fp32 + [bias_fp32]) / out_scale + out_zero_point
          = (act_uint8 - act_zero_point) * act_scale * (w_uint8 - w_zero_point)
* w_scale / out_scale + out_zero_point + [bias_fp32 / out_scale] = (act_uint8 *
w_uint8 - act_uint8 * w_zero_point - w_uint8 * act_zero_point
            + act_zero_point * w_zero_point) * act_scale * w_scale / out_scale +
            out_zero_point + [bias_fp32 / out_scale]
TODO(@fan.mo): mudnn only supports symmetric quantization without zero-point(=0)
          = (act_int8 * w_uint8 + [bias_fp32/(act_scale * w_scale)])
            * act_scale * w_scale / out_scale
          = (act_int8 * w_uint8 + [bias_fp32/(act_scale * w_scale)])
            / (out_scale / (act_scale * w_scale))
          = dequantize((act_int8 * w_uint8 + [bias_fp32/(act_scale * w_scale)]),
                        out_scale / (act_scale * w_scale))
*/
template <int kSpatialDim>
template <ActMode act_mode>
at::Tensor PackedConvWeightMudnn<kSpatialDim>::apply_impl(
    const at::Tensor& act,
    const c10::optional<at::Tensor>& accum,
    double output_scale,
    int64_t output_zero_point) {
  // Convolution attributes
  const auto batch_size = kSpatialDim == 2 ? act.size(0) : 1;
  const auto num_input_channels = act.size(kSpatialDim - 1);
  const auto H = act.size(kSpatialDim);
  const auto W = act.size(kSpatialDim + 1);
  const auto num_output_channels = weight_.size(0);
  std::vector<int64_t> kernel_size = {weight_.size(1), weight_.size(2)};
  at::ScalarType output_type = at::ScalarType::QInt8;
  at::SmallVector<int64_t, kSpatialDim + 2> output_shape =
      MakeQConvOutputShape<kSpatialDim>(
          batch_size,
          num_output_channels,
          {H, W},
          kernel_size,
          stride_,
          padding_,
          dilation_);
  at::Tensor quantized_output = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kPrivateUse1).dtype(output_type),
      output_scale,
      output_zero_point,
      c10::MemoryFormat::ChannelsLast);
  // permute to NHWC shape
  quantized_output = quantized_output.permute({0, 2, 3, 1});

  apply_impl_helper<act_mode>(
      quantized_output,
      act.to(c10::MemoryFormat::ChannelsLast),
      accum,
      output_scale,
      output_zero_point);

  if (output_channels_ != num_output_channels) {
    return quantized_output.slice(1, 0, output_channels_);
  }
  // permute back to NCHW shape
  quantized_output = quantized_output.permute({0, 3, 1, 2});

  return quantized_output;
}

template <int kSpatialDim>
at::Tensor PackedConvWeightMudnn<kSpatialDim>::apply(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<ActMode::IDENTITY>(
      input, c10::nullopt, output_scale, output_zero_point);
}

template <int kSpatialDim>
at::Tensor PackedConvWeightMudnn<kSpatialDim>::apply_relu(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<ActMode::RELU>(
      input, c10::nullopt, output_scale, output_zero_point);
}

template <int kSpatialDim>
at::Tensor PackedConvWeightMudnn<kSpatialDim>::apply_silu(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<ActMode::SILU>(
      input, c10::nullopt, output_scale, output_zero_point);
}

template at::Tensor PackedConvWeightMudnn<2>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightMudnn<2>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightMudnn<2>::apply_silu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

namespace at {
namespace musa {
namespace {

template <ActMode act_mode>
class QConvInt8 final {
 public:
  static at::Tensor run(
      at::Tensor act,
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    c10::musa::MUSAGuard device_guard(act.device());
    if (ActMode::RELU == act_mode) {
      return packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else if (ActMode::SILU == act_mode) {
      return dynamic_cast<PackedConvWeightMudnn<2>*>(packed_weight.get())
          ->apply_silu(act, output_scale, output_zero_point);
    } else if (ActMode::IDENTITY == act_mode) {
      return packed_weight->apply(act, output_scale, output_zero_point);
    } else {
      TORCH_CHECK(false, "Unsupported fusion activation");
    }
  }
};

TORCH_LIBRARY_IMPL(quantized, AutogradPrivateUse1, m) {
  // this is inconsistent with what has been done for conv2d where new variants
  // use packed weights, and old variant does not. we adopt this inconsistency
  // for now to be consistent with QuantizedCPU's conv1d and will eventually
  // deprecate the old variants
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv2d.new"),
      TORCH_FN(QConvInt8<ActMode::IDENTITY>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv2d_relu.new"),
      TORCH_FN(QConvInt8<ActMode::RELU>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv2d_silu.new"),
      TORCH_FN(QConvInt8<ActMode::SILU>::run));
}

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv2d.new"),
      TORCH_FN(QConvInt8<ActMode::IDENTITY>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv2d_relu.new"),
      TORCH_FN(QConvInt8<ActMode::RELU>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv2d_silu.new"),
      TORCH_FN(QConvInt8<ActMode::SILU>::run));
}

} // namespace
} // namespace musa
} // namespace at
