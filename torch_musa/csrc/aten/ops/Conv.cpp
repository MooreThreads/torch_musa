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
#include <ATen/ops/_slow_conv2d_backward.h>
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
#include <ATen/ops/miopen_convolution.h>
#include <ATen/ops/miopen_convolution_transpose.h>
#include <ATen/ops/miopen_depthwise_convolution.h>
#include <ATen/ops/mkldnn_convolution.h>
#include <ATen/ops/mps_convolution_backward.h>
#include <ATen/ops/mps_convolution_transpose_backward.h>
#include <ATen/ops/slow_conv3d.h>
#include <ATen/ops/slow_conv_dilated2d.h>
#include <ATen/ops/slow_conv_dilated3d.h>
#include <ATen/ops/slow_conv_transpose2d.h>
#include <ATen/ops/slow_conv_transpose3d.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/thnn_conv2d.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif
#include <ATen/native/ConvUtils.h>
#include <mudnn.h>
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace musa {

void Conv2dShapeCheck(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef padding,
    int64_t groups) {
  // Check dimension
  TORCH_CHECK(
      input.dim() == 4 && weight.dim() == 4,
      "Expected 4D for input and weight tensor");

  auto input_shape = input.sizes();
  auto weight_shape = weight.sizes();

  int64_t exactinput_height = input_shape[2] + 2 * padding[0];
  int64_t exactinput_width = input_shape[3] + 2 * padding[1];

  // Check input shape
  TORCH_CHECK(
      exactinput_height >= weight_shape[2] &&
          exactinput_width >= weight_shape[3],
      "input tensor should be greater than weight tensor!");
  // Check channel
  TORCH_CHECK(
      input_shape[1] / groups == weight_shape[1],
      "input.shape[1] should be the same with weight.shape[1]");
}

void Conv3dShapeCheck(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef padding,
    int64_t groups) {
  // Check dimension
  TORCH_CHECK(
      input.dim() == 5 && weight.dim() == 5,
      "Expected 5D for input and weight tensor");

  auto input_shape = input.sizes();
  auto weight_shape = weight.sizes();

  int64_t exactinput_depth = input_shape[2] + 2 * padding[0];
  int64_t exactinput_height = input_shape[3] + 2 * padding[1];
  int64_t exactinput_width = input_shape[4] + 2 * padding[2];

  // Check input shape
  TORCH_CHECK(
      exactinput_depth >= weight_shape[2] &&
          exactinput_height >= weight_shape[3] &&
          exactinput_width >= weight_shape[4],
      "input tensor should be greater than weight tensor!");
  // Check channel
  TORCH_CHECK(
      input_shape[1] / groups == weight_shape[1],
      "input.shape[1] should be the same with weight.shape[1]");
}

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

Tensor Conv2d(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  const auto weight_memory_format = weight.suggest_memory_format();
  Conv2dShapeCheck(input, weight, padding, groups);
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

  ::musa::dnn::Convolution::FusedActivationDesc act;
  act.SetMode(::musa::dnn::Convolution::FusedActivationDesc::Mode::IDENTITY);

  ::musa::dnn::Convolution::Algorithm algo =
      static_cast<::musa::dnn::Convolution::Algorithm>(0);
  c.GetRecommendForwardAlgorithm(h, algo, out, in, ke);

  at::musa::muTensor bias;
  if (bias_opt.has_value() && bias_opt.value().numel() != 0) {
    // FIXME: Will dangling pointer happen here? contiguous_bias may have its
    // own data area.
    bias = CreateMUTensor(bias_opt.value().contiguous());
  } else {
    bias = CreateEmptyMUTensor();
  }

  muTensor add = at::musa::CreateEmptyMUTensor();

  CHECK_MUDNN_STATUS(
      c.RunFusion(h, out, in, ke, bias, add, act, algo, InternalMemAlloc),
      "RunFusion");
  return output;
}

Tensor Conv2dTranspose(
    const Tensor& grad_output,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    int64_t groups,
    IntArrayRef dilation) {
  const auto weight_memory_format = weight.suggest_memory_format();
  auto input_size = at::native::conv_input_size(
      grad_output.sizes(),
      weight.sizes(),
      padding,
      output_padding,
      stride,
      dilation,
      groups);
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
      c.RunBwdData(h, gin, gout, w, algo, InternalMemAlloc), "RunBwdData");

  if (bias_opt.has_value()) {
    AddBias(grad_input_t, *bias_opt);
  }
  return grad_input_t;
}

Tensor Conv3d(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  Conv3dShapeCheck(input, weight, padding, groups);

  const auto weight_memory_format = weight.suggest_memory_format();
  Tensor contiguous_input = FormatContiguous(input, weight_memory_format);
  Tensor contiguous_weight = FormatContiguous(weight, weight_memory_format);

  auto input_shape = contiguous_input.sizes();
  auto weight_shape = contiguous_weight.sizes();

  auto output_size = at::native::conv_output_size(
      input_shape, weight_shape, padding, stride, dilation);
  Tensor output =
      at::empty(output_size, contiguous_input.options(), weight_memory_format);

  auto in = CreateMUTensor(contiguous_input);
  auto out = CreateMUTensor(output);
  auto ke = CreateMUTensor(contiguous_weight);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Convolution c;
  ConfigConv(c, input.scalar_type(), stride, padding, dilation, groups);
  ::musa::dnn::Convolution::Algorithm algo;
  c.GetRecommendForwardAlgorithm(h, algo, out, in, ke);
  if (bias_opt.has_value() && bias_opt.value().numel() != 0) {
#ifdef ENABLE_FUSION
    ::musa::dnn::Convolution::FusedActivationDesc fused_desc;
    muTensor temp_add;
    auto bias = CreateMUTensor(bias_opt.value().contiguous());
    CHECK_MUDNN_STATUS(
        c.RunFusion(
            h, out, in, ke, bias, temp_add, fused_desc, algo, InternalMemAlloc),
        "RunFusion");
#else
    CHECK_MUDNN_STATUS(
        c.Run(h, out, in, ke, algo, InternalMemAlloc), "RUNConv");
    if (bias_opt.has_value()) {
      AddBias(output, *bias_opt);
    }
#endif
  } else {
    // conv3d
    CHECK_MUDNN_STATUS(c.Run(h, out, in, ke, algo, InternalMemAlloc), "Run");
  }
  return output;
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
  TORCH_CHECK(
      weight.device().type() == kMUSA,
      "Device of weight tensor of Convolution must be MUSA, ",
      "but now is ",
      weight.device());
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of weight tensor of Convolution must be MUSA, but now is",
      input.device());
  TORCH_CHECK(
      weight.scalar_type() == at::ScalarType::Float ||
          weight.scalar_type() == at::ScalarType::Half,
      "Dtype of weight tensor of Convolution only support Float32 and Half, ",
      "but now it is ",
      weight.scalar_type());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float ||
          input.scalar_type() == at::ScalarType::Half,
      "Dtype of input tensor of Convolution only support Float32 and Half, ",
      "but now it is ",
      input.scalar_type());
  TORCH_CHECK(
      input.scalar_type() == weight.scalar_type(),
      "input dtype and weight dtype must be the same");
  c10::musa::MUSAGuard device_guard(input.device());
  if (input.dim() == 4 && weight.dim() == 4) {
    return transposed
        ? Conv2dTranspose(
              input,
              weight,
              bias_opt,
              stride,
              padding,
              output_padding,
              groups,
              dilation)
        : Conv2d(input, weight, bias_opt, stride, padding, dilation, groups);
  }
  if (input.dim() == 5 && weight.dim() == 5) {
    TORCH_CHECK(!transposed, "ConvTranposed3D is not supported on musa");
    return Conv3d(input, weight, bias_opt, stride, padding, dilation, groups);
  }
  auto input_host = input.to("cpu");
  auto weight_host = weight.to("cpu");
  c10::optional<Tensor> bias_host;
  if (bias_opt.has_value() && bias_opt.value().numel() != 0) {
    bias_host = bias_opt.value().to("cpu");
  }
  auto result_host = at::convolution(
      input_host,
      weight_host,
      bias_host,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups);
  return result_host.to(input.device());
}

Tensor Conv3dDataBwd(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    int64_t groups,
    IntArrayRef dilation) {
  const MUSAGuard device_guard(grad_output.device());
  auto grad_input_t = at::empty(input.sizes(), grad_output.options());

  Tensor contiguous_weight = weight.contiguous();
  Tensor contiguous_grad_output = grad_output.contiguous();

  auto gout = CreateMUTensor(contiguous_grad_output);
  auto gin = CreateMUTensor(grad_input_t);
  auto w = CreateMUTensor(contiguous_weight);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Convolution c;
  ConfigConv(c, input.scalar_type(), stride, padding, dilation, groups);
  ::musa::dnn::Convolution::AlgorithmBwdData algo;
  c.GetRecommendBackwardDataAlgorithm(h, algo, gin, gout, w);
  CHECK_MUDNN_STATUS(
      c.RunBwdData(h, gin, gout, w, algo, InternalMemAlloc), "ConvBwdData");
  return grad_input_t;
}

Tensor Conv2dDataBwd(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
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
      input.sizes(), contiguous_grad_output.options(), weight_memory_format);

  auto gout = CreateMUTensor(contiguous_grad_output);
  auto gin = CreateMUTensor(grad_input_t);
  auto w = CreateMUTensor(contiguous_weight);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Convolution c;
  ConfigConv(c, input.scalar_type(), stride, padding, dilation, groups);
  ::musa::dnn::Convolution::AlgorithmBwdData algo;
  c.GetRecommendBackwardDataAlgorithm(h, algo, gin, gout, w);
  CHECK_MUDNN_STATUS(
      c.RunBwdData(h, gin, gout, w, algo, InternalMemAlloc), "ConvBwdData");
  return grad_input_t;
}

Tensor Conv1dDataBwd(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    int64_t groups,
    IntArrayRef dilation) {
  Tensor contiguous_input = input.contiguous().unsqueeze(-1);
  Tensor contiguous_weight = weight.contiguous().unsqueeze(-1);
  Tensor contiguous_grad_output = grad_output.contiguous().unsqueeze(-1);

  std::vector<int64_t> vstride({stride[0], 1});
  std::vector<int64_t> vpadding({padding[0], 0});
  std::vector<int64_t> vdilation({dilation[0], 1});
  std::vector<int64_t> voutput_padding({output_padding[0], 0});
  auto result = Conv2dDataBwd(
      contiguous_grad_output,
      contiguous_input,
      contiguous_weight,
      vstride,
      vpadding,
      groups,
      vdilation);
  return result.squeeze_(-1);
}

Tensor Conv3dWeightBwd(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    int64_t groups,
    IntArrayRef dilation) {
  c10::musa::MUSAGuard device_guard(grad_output.device());
  auto weight_size = weight.sizes();
  auto grad_weight_t = at::empty(weight_size, grad_output.options());

  Tensor contiguous_input = input.contiguous();
  Tensor contiguous_grad_output = grad_output.contiguous();

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

Tensor Conv2dWeightBwd(
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

Tensor Conv1dWeightBwd(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    int64_t groups,
    IntArrayRef dilation) {
  TORCH_CHECK(
      weight.dim() == 3 && input.dim() == 3,
      "Expected 3D for weight tensor and input.");

  Tensor contiguous_input = input.contiguous().unsqueeze(-1);
  Tensor contiguous_weight = weight.contiguous().unsqueeze(-1);
  Tensor contiguous_grad_output = grad_output.contiguous().unsqueeze(-1);

  std::vector<int64_t> vstride({stride[0], 1});
  std::vector<int64_t> vpadding({padding[0], 0});
  std::vector<int64_t> vdilation({dilation[0], 1});
  Tensor grad_weight = Conv2dWeightBwd(
      contiguous_grad_output,
      contiguous_input,
      contiguous_weight,
      vstride,
      vpadding,
      groups,
      vdilation);
  return grad_weight.squeeze_(-1);
}

::std::tuple<Tensor, Tensor, Tensor> Convolution2dBwd(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    ::std::array<bool, 3> output_mask) {
  TORCH_CHECK(
      weight.dim() == 4 && input.dim() == 4,
      "Expected 4D for weight tensor and input.");
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  if (output_mask[0]) {
    grad_input = Conv2dDataBwd(
        grad_output, input, weight, stride, padding, groups, dilation);
  }

  if (output_mask[1]) {
    grad_weight = Conv2dWeightBwd(
        grad_output, input, weight, stride, padding, groups, dilation);
  }

  if (output_mask[2]) {
    grad_bias = at::sum(grad_output, IntArrayRef{0, 2, 3});
  }

  return std::tuple<Tensor&, Tensor&, Tensor&>{
      grad_input, grad_weight, grad_bias};
}

::std::tuple<Tensor, Tensor, Tensor> Convolution3dBwd(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    ::std::array<bool, 3> output_mask) {
  TORCH_CHECK(
      weight.dim() == 5 && input.dim() == 5,
      "Expected 5D for weight tensor and input.");
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  if (output_mask[0]) {
    grad_input = Conv3dDataBwd(
        grad_output, input, weight, stride, padding, groups, dilation);
  }

  if (output_mask[1]) {
    grad_weight = Conv3dWeightBwd(
        grad_output, input, weight, stride, padding, groups, dilation);
  }

  if (output_mask[2]) {
    grad_bias = at::sum(grad_output, IntArrayRef{0, 2, 3, 4});
  }

  return std::tuple<Tensor&, Tensor&, Tensor&>{
      grad_input, grad_weight, grad_bias};
}

::std::tuple<Tensor, Tensor, Tensor> Convolution1dBwd(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    IntArrayRef output_padding,
    int64_t groups,
    ::std::array<bool, 3> output_mask) {
  TORCH_CHECK(
      weight.dim() == 3 && input.dim() == 3,
      "Expected 3D for weight tensor and input.");
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  if (output_mask[0]) {
    grad_input = Conv1dDataBwd(
        grad_output,
        input,
        weight,
        stride,
        padding,
        output_padding,
        groups,
        dilation);
  }

  if (output_mask[1]) {
    grad_weight = Conv1dWeightBwd(
        grad_output, input, weight, stride, padding, groups, dilation);
  }

  if (output_mask[2]) {
    grad_bias = at::sum(grad_output, IntArrayRef{0, 2});
  }

  return std::tuple<Tensor&, Tensor&, Tensor&>{
      grad_input, grad_weight, grad_bias};
}

::std::tuple<Tensor, Tensor, Tensor> Conv2dTransposeBwd(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    int64_t groups,
    ::std::array<bool, 3> output_mask) {
  TORCH_CHECK(
      weight.dim() == 4 && input.dim() == 4,
      "Expected 4D for weight tensor and input.");
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  if (output_mask[0]) {
    Tensor bias_opt = at::zeros({weight.size(0)}, input.options());
    grad_input = Conv2d(
        grad_output, weight, bias_opt, stride, padding, dilation, groups);
  }
  if (output_mask[1]) {
    grad_weight = Conv2dWeightBwd(
        input, grad_output, weight, stride, padding, groups, dilation);
  }
  if (output_mask[2]) {
    grad_bias = at::sum(grad_output, IntArrayRef{0, 2, 3});
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
  TORCH_CHECK(
      weight.device().type() == kMUSA,
      "Device of weight tensor of Convolution Backward must be MUSA, ",
      "but now is ",
      weight.device());
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of weight tensor of Convolution Backward must be MUSA, "
      "but now is",
      input.device());
  TORCH_CHECK(
      grad_output.device().type() == kMUSA,
      "Device of grad_output tensor of Convolution Backward must be "
      "MUSA, but now is",
      grad_output.device());
  TORCH_CHECK(
      weight.scalar_type() == at::ScalarType::Float ||
          weight.scalar_type() == at::ScalarType::Half,
      "Dtype of weight tensor of Convolution Backward only support Float32 and Float16, ",
      "but now is ",
      weight.scalar_type());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float ||
          input.scalar_type() == at::ScalarType::Half,
      "Dtype of input tensor of Convolution Backward only support Float32 and Float16, ",
      "but now is ",
      input.scalar_type());
  TORCH_CHECK(
      grad_output.scalar_type() == at::ScalarType::Float ||
          grad_output.scalar_type() == at::ScalarType::Half,
      "Dtype of grad_output tensor of Convolution Backward only support Float32 and Float16, ",
      "but now is ",
      grad_output.scalar_type());
  c10::musa::MUSAGuard device_guard(input.device());

  if (input.dim() == 3 && weight.dim() == 3) {
    return Convolution1dBwd(
        grad_output,
        input,
        weight,
        stride,
        padding,
        dilation,
        output_padding,
        groups,
        output_mask);
  } else if (input.dim() == 4 && weight.dim() == 4) {
    return transposed ? Conv2dTransposeBwd(
                            grad_output,
                            input,
                            weight,
                            stride,
                            padding,
                            output_padding,
                            dilation,
                            groups,
                            output_mask)
                      : Convolution2dBwd(
                            grad_output,
                            input,
                            weight,
                            stride,
                            padding,
                            dilation,
                            groups,
                            output_mask);
  }
  return Convolution3dBwd(
      grad_output,
      input,
      weight,
      stride,
      padding,
      dilation,
      groups,
      output_mask);
}

ADVANCED_REGISTER(aten, PrivateUse1, "convolution_overrideable", Convolution)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "convolution_backward_overrideable",
    ConvolutionBwd)

} // namespace musa
} // namespace at
