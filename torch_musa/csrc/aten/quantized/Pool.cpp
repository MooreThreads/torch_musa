#include <c10/util/ArrayRef.h>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/quantized/mudnn/Conv.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <vector>

#include <mudnn.h>

at::SmallVector<int64_t, 4> MakePool2dOutputShape(
    int N, // batch size
    int M, // output channels
    const std::array<int64_t, 2>& input_sizes,
    at::IntArrayRef kernel,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation) {
  // make output shape of NCHW format
  const int H = input_sizes[0];
  const int W = input_sizes[1];
  const int64_t Y_H =
      (H + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1;
  const int64_t Y_W =
      (W + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1;
  return {N, Y_H, Y_W, M}; // NHWC
}

void QuantizedMaxPool2dImpl(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    const at::Tensor& inds,
    const double scale,
    const int64_t zero_point) {
  // mudnn handle and op
  at::musa::muHandle& h = at::GetMudnnHandle();
  ::musa::dnn::Pooling op;

  // op settings
  CHECK_MUDNN_STATUS(
      op.SetMode(::musa::dnn::Pooling::Mode::MAXPOOL), "Set MaxPooling mode");
  std::vector<int> kernel = {
      static_cast<int>(kernel_size[0]), static_cast<int>(kernel_size[1])};
  std::vector<int> pad_ = {
      static_cast<int>(padding[0]), static_cast<int>(padding[1])};
  std::vector<int> stride_ = {
      static_cast<int>(stride[0]), static_cast<int>(stride[1])};
  std::vector<int> dilation_ = {
      static_cast<int>(dilation[0]), static_cast<int>(dilation[1])};
  CHECK_MUDNN_STATUS(
      op.SetNdInfo(
          static_cast<int>(kernel_size.size()),
          kernel.data(),
          pad_.data(),
          stride_.data(),
          dilation_.data()),
      "Set MaxPooling Nd info");

  // muTensor create and settings
  at::Tensor input_ = input.permute({0, 2, 3, 1}); // permute to NHWC shape
  at::musa::muTensor mu_inds = at::musa::CreateMUTensor(inds);
  at::musa::muTensor mu_input = at::musa::CreateMUTensor(input_);
  at::musa::muTensor mu_out = at::musa::CreateMUTensor(output);
  SetMudnnQuantizationInfo(mu_input, scale, zero_point);
  SetMudnnQuantizationInfo(mu_out, scale, zero_point);
  CHECK_MUDNN_STATUS(
      mu_input.SetFormat(at::musa::muTensor::Format::NHWC),
      "Set input muTensor format as NHWC");
  CHECK_MUDNN_STATUS(
      mu_inds.SetFormat(at::musa::muTensor::Format::NHWC),
      "Set inds muTensor format as NHWC");
  CHECK_MUDNN_STATUS(
      mu_out.SetFormat(at::musa::muTensor::Format::NHWC),
      "Set output muTensor format as NHWC");

  // run kernel
  CHECK_MUDNN_STATUS(op.Run(h, mu_out, mu_input, mu_inds), "Run MaxPool2d");
}

void QuantizedAvgPool2dImpl(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& inds,
    const double scale,
    const int64_t zero_point) {
  // mudnn handle and op
  at::musa::muHandle& h = at::GetMudnnHandle();
  ::musa::dnn::Pooling op;

  // op settings
  CHECK_MUDNN_STATUS(
      op.SetMode(::musa::dnn::Pooling::Mode::ADAPTIVE_AVGPOOL),
      "Set AdaptiveAvgPooling mode");

  // muTensor create and settings
  at::Tensor input_ = input.permute({0, 2, 3, 1}); // permute to NHWC shape
  at::musa::muTensor mu_inds = at::musa::CreateMUTensor(inds);
  at::musa::muTensor mu_input = at::musa::CreateMUTensor(input_);
  at::musa::muTensor mu_out = at::musa::CreateMUTensor(output);
  SetMudnnQuantizationInfo(mu_input, scale, zero_point);
  SetMudnnQuantizationInfo(mu_out, scale, zero_point);
  CHECK_MUDNN_STATUS(
      mu_input.SetFormat(at::musa::muTensor::Format::NHWC),
      "Set input muTensor format as NHWC");
  CHECK_MUDNN_STATUS(
      mu_inds.SetFormat(at::musa::muTensor::Format::NHWC),
      "Set inds muTensor format as NHWC");
  CHECK_MUDNN_STATUS(
      mu_out.SetFormat(at::musa::muTensor::Format::NHWC),
      "Set output muTensor format as NHWC");

  // run kernel
  CHECK_MUDNN_STATUS(op.Run(h, mu_out, mu_input, mu_inds), "Run MaxPool2d");
}

namespace at {
namespace musa {

Tensor AdaptiveAvgPool2dQuantized(
    const at::Tensor& input,
    IntArrayRef output_size) {
  const OptionalDeviceGuard device_guard(device_of(input));
  TORCH_CHECK(
      input.qscheme() == at::kPerTensorAffine,
      "AdaptiveAvgPool2dQuantized only supports per tensor quantized tensors");
  TORCH_CHECK(
      output_size.size() == 2,
      "AdaptiveAvgPool2d only supports output with 2-elements");

  double scale = input.q_scale();
  int64_t zero_point = input.q_zero_point();

  at::Tensor output = at::_empty_affine_quantized(
      {input.size(0), output_size[0], output_size[1], input.size(1)},
      at::device(at::kPrivateUse1).dtype(input.scalar_type()),
      scale,
      zero_point,
      c10::MemoryFormat::Contiguous);
  at::Tensor inds = at::empty(
      {input.size(0), output_size[0], output_size[1], input.size(1)},
      at::device(at::kPrivateUse1)
          .dtype(at::ScalarType::Long)
          .memory_format(c10::MemoryFormat::Contiguous));

  QuantizedAvgPool2dImpl(
      output,
      input.to(c10::MemoryFormat::ChannelsLast),
      inds,
      scale,
      zero_point);

  return output.permute({0, 3, 1, 2});
}

Tensor MaxPool2dQuantized(
    const at::Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  const OptionalDeviceGuard device_guard(device_of(input));
  (void)ceil_mode; // disable unused variable lint warning
  if (stride.empty()) {
    stride = kernel_size;
  }
  auto ndim = input.dim();
  TORCH_CHECK(
      input.qscheme() == at::kPerTensorAffine,
      "MaxPool2dQuantized(): only supports per tensor quantized tensors");
  TORCH_CHECK(
      ndim == 3 || ndim == 4, "Expecting the input tensor of rank 3 or 4.");
  TORCH_CHECK(
      kernel_size.size() == 2,
      "MaxPool2dQuantized(): Expected kernel_size to be 2-dimensional: got ",
      kernel_size.size());
  TORCH_CHECK(
      stride.size() == 2,
      "MaxPool2dQuantized(): Expected stride to be 2-dimensional: got ",
      stride.size());
  TORCH_CHECK(
      dilation.size() == 2,
      "MaxPool2dQuantized(): Expected dilation to be 2-dimensional: got ",
      dilation.size());
  TORCH_CHECK(
      dilation[0] == 1 && dilation[1] == 1,
      "MaxPool2dQuantized(): Expected dilation=[1, 1] (cudnn does not currently support dilation[i] != 1), got",
      dilation);
  TORCH_CHECK(
      padding.size() == 2,
      "MaxPool2dQuantized(): Expected padding to be 2-dimensional: got ",
      padding.size());
  TORCH_CHECK(
      input.scalar_type() == c10::kQInt8,
      "MaxPool2dQuantized(): Expected dataype to be qint8, got ",
      input.scalar_type());

  double scale = input.q_scale();
  int64_t zero_point = input.q_zero_point();

  at::SmallVector<int64_t, 4> output_shape = MakePool2dOutputShape(
      input.size(0),
      input.size(1),
      {input.size(2), input.size(3)},
      kernel_size,
      stride,
      padding,
      dilation); // NHWC shape
  at::Tensor output = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kPrivateUse1).dtype(input.scalar_type()),
      scale,
      zero_point,
      c10::MemoryFormat::Contiguous); // no need for ChannelsLast format since
                                      // we already have NHWC shape
  at::Tensor inds = at::empty(
      output_shape,
      at::device(at::kPrivateUse1)
          .dtype(at::ScalarType::Long)
          .memory_format(c10::MemoryFormat::Contiguous));

  QuantizedMaxPool2dImpl(
      output,
      input.to(c10::MemoryFormat::ChannelsLast),
      kernel_size,
      stride,
      padding,
      dilation,
      inds,
      scale,
      zero_point);

  return output.permute({0, 3, 1, 2});
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("_adaptive_avg_pool2d", TORCH_FN(AdaptiveAvgPool2dQuantized));
  m.impl("quantized_max_pool2d", TORCH_FN(MaxPool2dQuantized));
}
} // namespace musa
} // namespace at
