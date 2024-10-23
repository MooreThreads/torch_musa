#include <algorithm>

#include <ATen/Config.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_conv_depthwise2d.h>
#include <ATen/ops/_convolution.h>
#include <ATen/ops/_convolution_double_backward_native.h>
#include <ATen/ops/_convolution_mode.h>
#include <ATen/ops/_convolution_mode_native.h>
#include <ATen/ops/_convolution_native.h>
#include <ATen/ops/_unsafe_view.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/conv1d_native.h>
#include <ATen/ops/conv2d_native.h>
#include <ATen/ops/conv3d_native.h>
#include <ATen/ops/conv_depthwise3d.h>
#include <ATen/ops/conv_transpose1d_native.h>
#include <ATen/ops/conv_transpose2d_native.h>
#include <ATen/ops/conv_transpose3d_native.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/convolution_backward_native.h>
#include <ATen/ops/convolution_backward_overrideable.h>
#include <ATen/ops/convolution_backward_overrideable_native.h>
#include <ATen/ops/convolution_native.h>
#include <ATen/ops/convolution_overrideable.h>
#include <ATen/ops/convolution_overrideable_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif
#include <ATen/native/ConvUtils.h>
#include <mudnn.h>
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {
void ConfigConv(
    ::musa::dnn::Convolution& c,
    const at::ScalarType& dtype,
    IntArrayRef str,
    IntArrayRef pad,
    IntArrayRef dil,
    int64_t groups) {
  auto sz = str.size();
  if (sz == 2) {
    CHECK_MUDNN_STATUS(
        c.SetNdInfo(
            {static_cast<int>(pad[0]), static_cast<int>(pad[1])},
            {static_cast<int>(str[0]), static_cast<int>(str[1])},
            {static_cast<int>(dil[0]), static_cast<int>(dil[1])}),
        "SetNdInfo");
  } else {
    // conv3d
    CHECK_MUDNN_STATUS(
        c.SetNdInfo(
            {static_cast<int>(pad[0]),
             static_cast<int>(pad[1]),
             static_cast<int>(pad[2])},
            {static_cast<int>(str[0]),
             static_cast<int>(str[1]),
             static_cast<int>(str[2])},
            {static_cast<int>(dil[0]),
             static_cast<int>(dil[1]),
             static_cast<int>(dil[2])}),
        "SetNdInfo");
  }
  CHECK_MUDNN_STATUS(
      c.SetComputeMode(at::musa::GetComputeModeFromCtx(dtype)),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(c.SetGroups(groups), "SetGroups");
}

void AddBias(Tensor& out, const Tensor& bias) {
  if (bias.numel() == 0) {
    return;
  }
  TORCH_CHECK(bias.dim() == 1, "Dimension of bias should be 1");
  out.add_(at::native::reshape_bias(out.dim(), bias));
}

template <int N>
Tensor ConvNd(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  static_assert(N == 2 || N == 3);
  const auto weight_memory_format = weight.suggest_memory_format();

  // There is no need to check shape, see
  // https://github.com/pytorch/pytorch/blob/39901f229520a5256505ec24782f716ee7ddc843/aten/src/ATen/native/Convolution.cpp#L1515

  Tensor contiguous_input = FormatContiguous(input, weight_memory_format);
  Tensor contiguous_weight = FormatContiguous(weight, weight_memory_format);

  auto input_shape = contiguous_input.sizes();
  auto weight_shape = contiguous_weight.sizes();

  Tensor output;
  auto output_size = at::native::conv_output_size(
      input_shape, weight_shape, padding, stride, dilation);
  output =
      at::empty(output_size, contiguous_input.options(), weight_memory_format);
  auto in = CreateMUTensor(contiguous_input);
  auto out = CreateMUTensor(output);
  auto ke = CreateMUTensor(contiguous_weight);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Convolution c;
  ConfigConv(c, input.scalar_type(), stride, padding, dilation, groups);

  ::musa::dnn::Convolution::Algorithm algo;
  c.GetRecommendForwardAlgorithm(h, algo, out, in, ke);

  if constexpr (N == 2) {
    ::musa::dnn::Convolution::FusedActivationDesc act;
    act.SetMode(::musa::dnn::Convolution::FusedActivationDesc::Mode::IDENTITY);

    at::musa::muTensor bias;
    if (bias_opt.has_value() && bias_opt.value().numel() != 0) {
      // bias always contiguous ?
      bias = CreateMUTensor(bias_opt.value());
    } else {
      bias = CreateEmptyMUTensor();
    }
    muTensor add = at::musa::CreateEmptyMUTensor();
    CHECK_MUDNN_STATUS(
        c.RunFusion(h, out, in, ke, bias, add, act, algo, InternalMemAlloc),
        "RunFusion");
  } else {
    // conv3d
    CHECK_MUDNN_STATUS(c.Run(h, out, in, ke, algo, InternalMemAlloc), "Run");
    if (bias_opt.has_value()) {
      AddBias(output, *bias_opt);
    }
  }

  return output;
}

template <int ND>
Tensor ConvDataBwd(
    const Tensor&,
    const Tensor&,
    IntArrayRef,
    IntArrayRef,
    IntArrayRef,
    int64_t,
    IntArrayRef);

template <int ND>
Tensor ConvTransposeImpl(
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    int64_t groups,
    IntArrayRef dilation) {
  auto input_size = at::native::conv_input_size(
      grad_output.sizes(),
      weight.sizes(),
      padding,
      output_padding,
      stride,
      dilation,
      groups);

  return ConvDataBwd<ND>(
      grad_output, weight, input_size, stride, padding, groups, dilation);
}

template <int ND>
Tensor ConvTranspose(
    const Tensor& grad_output,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    int64_t groups,
    IntArrayRef dilation) {
  static_assert(ND == 2 || ND == 3);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(padding.size() == ND);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dilation.size() == ND);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(weight.dim() == (ND + 2));

  std::vector<int64_t> maybe_shrinked_padding(ND);
  std::vector<int64_t> padding_gap(ND * 2);
  bool not_shrink_padding = true;
  for (int i = 0; i < ND; i++) {
    const auto pad_i_upperbound = dilation[i] * (weight.size(i + 2) - 1);
    const auto pad_i_gap = std::min<int64_t>(pad_i_upperbound - padding[i], 0);
    const bool not_shrink_pad_i = (pad_i_gap == 0);
    maybe_shrinked_padding[i] =
        (not_shrink_pad_i ? padding[i] : pad_i_upperbound);
    padding_gap[ND * 2 - 1 - i * 2] = pad_i_gap;
    padding_gap[ND * 2 - 2 - i * 2] = pad_i_gap;
    not_shrink_padding = not_shrink_pad_i && not_shrink_padding;
  }

  Tensor grad_input_t = ConvTransposeImpl<ND>(
      grad_output,
      weight,
      stride,
      maybe_shrinked_padding,
      output_padding,
      groups,
      dilation);

  if (C10_UNLIKELY(!not_shrink_padding)) {
    grad_input_t = at::pad(grad_input_t, IntArrayRef(padding_gap));
  }
  if (bias_opt.has_value()) {
    AddBias(grad_input_t, *bias_opt);
  }
  return grad_input_t;
}

Tensor Convolution(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups) {
  // definitely on MUSA device
  TORCH_CHECK(
      weight.device() == input.device(),
      "Expected weight tensor and input tensor to be on the same MUSA device, ",
      "but got weight on ",
      weight.device(),
      " and input on ",
      input.device());
  TORCH_CHECK(
      weight.scalar_type() == at::ScalarType::Float ||
          weight.scalar_type() == at::ScalarType::Half ||
          weight.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of weight tensor of Convolution only support Float32 and Half/BFloat16, ",
      "but now it is ",
      weight.scalar_type());
  TORCH_CHECK(
      input.scalar_type() == weight.scalar_type(),
      "input dtype and weight dtype must be the same in Convolution");
  c10::musa::MUSAGuard device_guard(input.device());

  // only conv2d and conv3d, 3D tensors have been unsqueezed by native torch
  // before calling into Convolution
  if (input.dim() == 4 && weight.dim() == 4) {
    return transposed
        ? ConvTranspose<2>(
              input,
              weight,
              bias_opt,
              stride,
              padding,
              output_padding,
              groups,
              dilation)
        : ConvNd<2>(input, weight, bias_opt, stride, padding, dilation, groups);
  }
  return transposed
      ? ConvTranspose<3>(
            input,
            weight,
            bias_opt,
            stride,
            padding,
            output_padding,
            groups,
            dilation)
      : ConvNd<3>(input, weight, bias_opt, stride, padding, dilation, groups);
}

static muTensor CreateMUTensorFromNDHWCToDHWCN(
    const Tensor& out,
    const Tensor& in) {
  // called by ConvDataBwd in ChannelsLast3d case
  // NOTE: The memory format at Torch level makes no sense,
  // i.e., the out just needs to satisfy a dense layout
  muTensor in_mu, out_mu;

  SetMUTensorDType(out.scalar_type(), out_mu);
  SetMUTensorAddr(out.data_ptr(), out_mu);
  out_mu.SetFormat(muTensor::Format::DHWCN);
  auto sizes = in.sizes();
  // desired out's shape and stride:
  // shape: (sizes[0], sizes[1], sizes[2], sizes[3], sizes[4])
  // stride: (1,
  //          sizes[0],
  //          sizes[0] * sizes[1] * sizes[4] * sizes[3],
  //          sizes[0] * sizes[1] * sizes[4],
  //          sizes[0] * sizes[1])
  int64_t stride_4 = sizes[0] * sizes[1];
  int64_t stride_3 = stride_4 * sizes[4];
  int64_t stride_2 = stride_3 * sizes[3];

  out_mu.SetNdInfo(
      {sizes[2], sizes[3], sizes[4], sizes[1], sizes[0]},
      {stride_2, stride_3, stride_4, sizes[0], 1});

  SetMUTensorDType(in.scalar_type(), in_mu);
  SetMUTensorAddr(in.data_ptr(), in_mu);
  in_mu.SetFormat(muTensor::Format::NDHWC);
  // desired in's shape and stride:
  // shape: (sizes[0], sizes[1], sizes[2], sizes[3], sizes[4])
  // stride: (sizes[1] * sizes[2] * sizes[3] * sizes[4],
  //          1,
  //          sizes[1] * sizes[4] * sizes[3],
  //          sizes[1] * sizes[4],
  //          sizes[1])
  stride_3 = sizes[1] * sizes[4];
  stride_2 = stride_3 * sizes[3];
  int64_t stride_0 = stride_2 * sizes[2];
  in_mu.SetNdInfo(
      {sizes[2], sizes[3], sizes[4], sizes[1], sizes[0]},
      {stride_2, stride_3, sizes[1], 1, stride_0});

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Permute op;
  CHECK_MUDNN_STATUS(
      op.Run(h, out_mu, in_mu), "CreateMUTensorFromNDHWCToDHWCN");

  return out_mu;
}

template <int ND>
Tensor ConvDataBwd(
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef input_size,
    IntArrayRef stride,
    IntArrayRef padding,
    int64_t groups,
    IntArrayRef dilation) {
  c10::musa::MUSAGuard device_guard(grad_output.device());

  const auto weight_memory_format = weight.suggest_memory_format();
  Tensor contiguous_weight = FormatContiguous(weight, weight_memory_format);

  Tensor contiguous_grad_output =
      FormatContiguous(grad_output, weight_memory_format);
  auto grad_input_t = at::empty(
      input_size, contiguous_grad_output.options(), weight_memory_format);

  auto gout = CreateMUTensor(contiguous_grad_output);
  auto gin = CreateMUTensor(grad_input_t);
  auto w = CreateMUTensor(contiguous_weight);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Convolution c;
  ConfigConv(c, weight.scalar_type(), stride, padding, dilation, groups);
  ::musa::dnn::Convolution::AlgorithmBwdData algo;
  c.GetRecommendBackwardDataAlgorithm(h, algo, gin, gout, w);
  CHECK_MUDNN_STATUS(
      c.RunBwdData(h, gin, gout, w, algo, InternalMemAlloc), "ConvBwdData");
  return grad_input_t;
}

// Convolution::RunBwdData only support DHWCN format weight in NDHWC case
// that's why make a template specilization here, we could remove this once
// NDHWC weight is supported.
template <>
Tensor ConvDataBwd<3>(
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef input_size,
    IntArrayRef stride,
    IntArrayRef padding,
    int64_t groups,
    IntArrayRef dilation) {
  c10::musa::MUSAGuard device_guard(grad_output.device());
  const auto weight_memory_format = weight.suggest_memory_format();

  Tensor contiguous_weight;
  muTensor w;
  if (weight_memory_format == MemoryFormat::ChannelsLast3d) {
    // call FormatContiguous to ensure restride
    contiguous_weight = at::empty(weight.sizes(), weight.options());
    w = CreateMUTensorFromNDHWCToDHWCN(
        contiguous_weight, FormatContiguous(weight, weight_memory_format));

  } else {
    contiguous_weight = FormatContiguous(weight, weight_memory_format);
    w = CreateMUTensor(contiguous_weight);
  }

  Tensor contiguous_grad_output =
      FormatContiguous(grad_output, weight_memory_format);
  auto grad_input_t = at::empty(
      input_size, contiguous_grad_output.options(), weight_memory_format);

  auto gout = CreateMUTensor(contiguous_grad_output);
  auto gin = CreateMUTensor(grad_input_t);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Convolution c;
  ConfigConv(c, weight.scalar_type(), stride, padding, dilation, groups);
  ::musa::dnn::Convolution::AlgorithmBwdData algo;
  c.GetRecommendBackwardDataAlgorithm(h, algo, gin, gout, w);
  CHECK_MUDNN_STATUS(
      c.RunBwdData(h, gin, gout, w, algo, InternalMemAlloc), "ConvBwdData");
  return grad_input_t;
}

Tensor ConvWeightBwd(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    int64_t groups,
    IntArrayRef dilation) {
  c10::musa::MUSAGuard device_guard(grad_output.device());
  const auto weight_memory_format = weight.suggest_memory_format();
  Tensor contiguous_input = FormatContiguous(input, weight_memory_format);
  Tensor contiguous_grad_output =
      FormatContiguous(grad_output, weight_memory_format);

  auto weight_size = weight.sizes();
  auto grad_weight_t = at::empty(
      weight_size, contiguous_grad_output.options(), weight_memory_format);

  auto gout = CreateMUTensor(contiguous_grad_output);
  auto gw = CreateMUTensor(grad_weight_t);
  auto in = CreateMUTensor(contiguous_input);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Convolution c;
  ConfigConv(c, input.scalar_type(), stride, padding, dilation, groups);
  ::musa::dnn::Convolution::AlgorithmBwdFilter algo;
  c.GetRecommendBackwardFilterAlgorithm(h, algo, gw, in, gout);
  CHECK_MUDNN_STATUS(
      c.RunBwdFilter(h, gw, in, gout, algo, InternalMemAlloc), "ConvBwdFilter");
  return grad_weight_t;
}

template <int ND>
::std::tuple<Tensor, Tensor, Tensor> ConvolutionBwdImpl(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    ::std::array<bool, 3> output_mask) {
  static_assert(ND == 2 || ND == 3);
  Tensor grad_input, grad_weight, grad_bias;

  Tensor grad_output_c = grad_output;

  if constexpr (ND == 2) {
    const auto weight_memory_format = weight.suggest_memory_format();
    grad_output_c = FormatContiguous(grad_output, weight_memory_format);
  }

  if (output_mask[0]) {
    grad_input = ConvDataBwd<ND>(
        grad_output_c,
        weight,
        input.sizes(),
        stride,
        padding,
        groups,
        dilation);
  }

  if (output_mask[1]) {
    grad_weight = ConvWeightBwd(
        grad_output_c, input, weight, stride, padding, groups, dilation);
  }

  if (output_mask[2]) {
    if constexpr (ND == 2) {
      grad_bias = at::sum(grad_output, IntArrayRef{0, 2, 3});
    } else {
      grad_bias = at::sum(grad_output, IntArrayRef{0, 2, 3, 4});
    }
  }

  return std::tuple<Tensor&, Tensor&, Tensor&>{
      grad_input, grad_weight, grad_bias};
}

template <int N>
::std::tuple<Tensor, Tensor, Tensor> ConvTransposeBwd(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    int64_t groups,
    ::std::array<bool, 3> output_mask) {
  static_assert(N == 2 || N == 3);
  Tensor grad_input, grad_weight, grad_bias;

  const auto weight_memory_format = weight.suggest_memory_format();
  Tensor grad_output_c = FormatContiguous(grad_output, weight_memory_format);

  if (output_mask[0]) {
    grad_input = ConvNd<N>(
        grad_output_c,
        weight,
        c10::optional<Tensor>(),
        stride,
        padding,
        dilation,
        groups);
  }
  if (output_mask[1]) {
    Tensor input_c = FormatContiguous(input, weight_memory_format);
    grad_weight = ConvWeightBwd(
        input_c, grad_output_c, weight, stride, padding, groups, dilation);
  }
  if (output_mask[2]) {
    if constexpr (N == 2) {
      grad_bias = at::sum(grad_output, IntArrayRef{0, 2, 3});
    } else {
      grad_bias = at::sum(grad_output, IntArrayRef{0, 2, 3, 4});
    }
  }

  return std::tuple<Tensor&, Tensor&, Tensor&>{
      grad_input, grad_weight, grad_bias};
}

::std::tuple<Tensor, Tensor, Tensor> ConvolutionBwd(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    ::std::array<bool, 3> output_mask) {
  // definitely on MUSA device
  TORCH_CHECK(
      (weight.device() == input.device()) &&
          (weight.device() == grad_output.device()),
      "Expected weight tensor, input tensor and grad_output tensor ",
      "to be on the same MUSA device, but got weight on ",
      weight.device(),
      " input on ",
      input.device(),
      " and grad_output on ",
      grad_output.device());
  TORCH_CHECK(
      weight.scalar_type() == at::ScalarType::Float ||
          weight.scalar_type() == at::ScalarType::Half ||
          weight.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of weight tensor of Convolution Backward only support Float32 and Float16/Bfloat16, ",
      "but now is ",
      weight.scalar_type());
  TORCH_CHECK(
      (weight.scalar_type() == input.scalar_type()) &&
          (weight.scalar_type() == grad_output.scalar_type()),
      "input's dtype, weight's dtype and grad_output's dtype must be the same in ConvolutionBwd");
  c10::musa::MUSAGuard device_guard(input.device());

  // only conv2d and conv3d, 3D tensors have been unsqueezed by native torch
  // before calling into ConvolutionBwd
  if (input.dim() == 4 && weight.dim() == 4) {
    return transposed ? ConvTransposeBwd<2>(
                            grad_output,
                            input,
                            weight,
                            stride,
                            padding,
                            output_padding,
                            dilation,
                            groups,
                            output_mask)
                      : ConvolutionBwdImpl<2>(
                            grad_output,
                            input,
                            weight,
                            stride,
                            padding,
                            dilation,
                            groups,
                            output_mask);
  }
  return transposed ? ConvTransposeBwd<3>(
                          grad_output,
                          input,
                          weight,
                          stride,
                          padding,
                          output_padding,
                          dilation,
                          groups,
                          output_mask)
                    : ConvolutionBwdImpl<3>(
                          grad_output,
                          input,
                          weight,
                          stride,
                          padding,
                          dilation,
                          groups,
                          output_mask);
}

} // namespace musa
} // namespace at
