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

template <int kSpatialDim>
at::SmallVector<int64_t, kSpatialDim + 2> MakeQConvOutputShape(
    int N, // mini-batch
    int M, // output channels
    const std::array<int64_t, kSpatialDim>& input_image_shape,
    const std::vector<int64_t>& kernel,
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& padding,
    const torch::List<int64_t>& dilation);

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

void inline SetMudnnQuantizationInfo(
    at::musa::muTensor& self,
    double scales,
    int64_t zero_points) {
  float scales_ = static_cast<float>(scales);
  unsigned int zero_points_ = static_cast<unsigned int>(zero_points);
  CHECK_MUDNN_STATUS(
      self.SetQuantizationInfo(1, &scales_, &zero_points_),
      "Set quantization info");
}

void inline SetMudnnFormat(at::musa::muTensor& t) {
  CHECK_MUDNN_STATUS(
      t.SetFormat(at::musa::muTensor::Format::NHWC),
      "Set mudnn tensor format NHWC");
}

void inline ConfigConv(
    ::musa::dnn::Convolution& c,
    torch::List<int64_t> pad,
    torch::List<int64_t> str,
    torch::List<int64_t> dil) {
  CHECK_MUDNN_STATUS(
      c.SetGroups(1),
      "SetGroups"); // We currently only support groups == 1 qconv
  CHECK_MUDNN_STATUS(
      c.SetComputeMode(::musa::dnn::Convolution::ComputeMode::ALL),
      "SetComputeMode");

  int sizes = pad.size();
  if (sizes == 2) {
    std::vector<int> pad_ = {
        static_cast<int>(pad[0]), static_cast<int>(pad[1])};
    std::vector<int> str_ = {
        static_cast<int>(str[0]), static_cast<int>(str[1])};
    std::vector<int> dil_ = {
        static_cast<int>(dil[0]), static_cast<int>(dil[1])};
    CHECK_MUDNN_STATUS(
        c.SetNdInfo(sizes, pad_.data(), str_.data(), dil_.data()), "SetNdInfo");
  } else {
    TORCH_CHECK(false, "We currently only support quantized conv2d");
  }
}

template <int kSpatialDim>
template <bool kReluFused>
void PackedConvWeightMudnn<kSpatialDim>::apply_impl_helper(
    at::Tensor& quantized_output,
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  // scale(s) and zero_point(s) of activation and weight
  TORCH_CHECK(
      input.qscheme() == c10::kPerTensorAffine ||
          input.qscheme() == c10::kPerTensorSymmetric,
      "Conv2d only supports per-tensor quantized activation");

  auto act_scale = input.q_scale();
  auto act_zero_point = input.q_zero_point();
  auto weight_scale = maybe_padded_weight_.q_scale();
  auto weight_zero_point = maybe_padded_weight_.q_zero_point();

  // permute to NHWC format
  at::Tensor input_ = input.permute({0, 2, 3, 1});
  at::musa::muTensor in = at::musa::CreateMUTensor(input_);
  at::musa::muTensor ke = at::musa::CreateMUTensor(maybe_padded_weight_);
  at::musa::muTensor out = at::musa::CreateMUTensor(quantized_output);
  // set quantization info (scale and zero_point) to muTensors
  // and set their format
  SetMudnnFormat(in);
  SetMudnnFormat(out);
  SetMudnnFormat(ke);
  SetMudnnQuantizationInfo(in, act_scale, act_zero_point);
  SetMudnnQuantizationInfo(out, output_scale, output_zero_point);
  SetMudnnQuantizationInfo(ke, weight_scale, weight_zero_point);

  // if bias is used, we should make a muTensor and broadcast it first
  at::musa::muTensor bias;
  if (bias_.has_value() && bias_.value().numel() != 0) {
    bias = at::musa::CreateMUTensor(
        bias_.value()); // TODO(@fan.mo): check if broadcast is needed
  } else {
    bias = at::musa::CreateEmptyMUTensor();
  }

  // mudnn kernel support skip-connection add but we currently won't use that
  at::musa::muTensor add = at::musa::CreateEmptyMUTensor();

  at::musa::muHandle& h = at::GetMudnnHandle();
  ::musa::dnn::Convolution op;
  ConfigConv(op, padding(), stride(), dilation());

  // from mudnn, 2 stands for RELU and 0 stands for IDENTITY
  ::musa::dnn::Convolution::FusedActivationDesc act;
  if (kReluFused) {
    act.SetMode(
        static_cast<::musa::dnn::Convolution::FusedActivationDesc::Mode>(2));
  } else {
    act.SetMode(
        static_cast<::musa::dnn::Convolution::FusedActivationDesc::Mode>(0));
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

// Since mudnn(QY1) only supports uint8 format Tensor,
// weight and output Tensor will be a clampped uint8 Tensor
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
template <bool kReluFused>
at::Tensor PackedConvWeightMudnn<kSpatialDim>::apply_impl(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point) {
  // Convolution attributes
  const auto batch_size = kSpatialDim == 2 ? act.size(0) : 1;
  const auto num_input_channels = act.size(kSpatialDim - 1);
  const auto H = act.size(kSpatialDim);
  const auto W = act.size(kSpatialDim + 1);
  const auto num_output_channels = maybe_padded_weight_.size(3);
  std::vector<int64_t> kernel_size = {
      maybe_padded_weight_.size(0), maybe_padded_weight_.size(1)};

  at::SmallVector<int64_t, kSpatialDim + 2> output_shape =
      MakeQConvOutputShape<kSpatialDim>(
          batch_size,
          num_output_channels,
          {H, W},
          kernel_size,
          stride_,
          padding_,
          dilation_);
  // cudnn supports QInt8 output while mudnn only supports QUInt8 currently
  at::Tensor quantized_output = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kPrivateUse1).dtype(at::ScalarType::QUInt8),
      output_scale,
      output_zero_point,
      c10::MemoryFormat::ChannelsLast);

  // mudnn requires input_channel to be multiplier of 32,
  // would be modified to 4 or 8 later (cudnn is 4)
  auto act_maybe_padded = act;
  if (num_input_channels % 32 != 0) {
    int8_t num_slices =
        32 - num_input_channels % 32; // number of slices we need to pad
    act_maybe_padded =
        at::pad(act, {0, 0, 0, 0, 0, num_slices, 0, 0}, "constant", 0);
  }

  quantized_output = quantized_output.permute({0, 2, 3, 1});
  apply_impl_helper<kReluFused>(
      quantized_output,
      act_maybe_padded.to(c10::MemoryFormat::ChannelsLast),
      output_scale,
      output_zero_point);

  // permute back to NCHW format
  quantized_output = quantized_output.permute({0, 3, 1, 2});

  // need to return sliced tensor if output_channels was padded
  if (num_unpadded_output_channels_ != maybe_padded_weight_.size(3)) {
    return quantized_output.slice(1, 0, num_unpadded_output_channels_);
  }
  return quantized_output;
}

template <int kSpatialDim>
at::Tensor PackedConvWeightMudnn<kSpatialDim>::apply(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  c10::musa::MUSAGuard device_guard(input.device());
  return apply_impl<false>(input, output_scale, output_zero_point);
}

template <int kSpatialDim>
at::Tensor PackedConvWeightMudnn<kSpatialDim>::apply_relu(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  c10::musa::MUSAGuard device_guard(input.device());
  return apply_impl<true>(input, output_scale, output_zero_point);
}

template at::Tensor PackedConvWeightMudnn<2>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightMudnn<2>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

namespace at {
namespace musa {
namespace {

template <bool kReluFused>
class QConvInt8 final {
 public:
  static at::Tensor run(
      at::Tensor act,
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    c10::musa::MUSAGuard device_guard(act.device());
    if (kReluFused) {
      return packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      return packed_weight->apply(act, output_scale, output_zero_point);
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
      TORCH_FN(QConvInt8<false>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv2d_relu.new"),
      TORCH_FN(QConvInt8<true>::run));
}

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv2d.new"),
      TORCH_FN(QConvInt8<false>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv2d_relu.new"),
      TORCH_FN(QConvInt8<true>::run));
}

} // namespace
} // namespace musa
} // namespace at
