#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/Pool.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/adaptive_avg_pool2d.h>
#include <ATen/ops/dequantize.h>
#include <ATen/ops/max_pool2d_with_indices.h>
#include <ATen/ops/quantize_per_tensor.h>
#endif

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/quantized/QTensor.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <torch/library.h>

namespace at {
namespace musa {
// TODO: replace avg_pool2d and max_pool2d with a int8 musa implementation
Tensor AdaptiveAvgPool2dQuantized(
    const at::Tensor& input,
    IntArrayRef output_size) {
  const OptionalDeviceGuard device_guard(device_of(input));
  TORCH_CHECK(
      input.qscheme() == at::kPerTensorAffine,
      "AdaptiveAvgPool2dQuantized only supports per tensor quantized tensors");
  auto input_fp32 = at::dequantize(input);
  auto result_fp32 = at::adaptive_avg_pool2d(input_fp32, output_size);
  return at::quantize_per_tensor(
      result_fp32, input.q_scale(), input.q_zero_point(), input.scalar_type());
}

Tensor MaxPool2dQuantized(
    const at::Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  const OptionalDeviceGuard device_guard(device_of(input));
  if (stride.empty()) {
    stride = kernel_size;
  }
  auto ndim = input.dim();
  TORCH_CHECK(
      input.qscheme() == at::kPerTensorAffine,
      "adaptive_avg_pool2d_quantized_cuda only supports per tensor quantized tensors");
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

  auto input_fp32 = at::dequantize(input);

  auto [result_fp32, result_indice] = at::max_pool2d_with_indices(
      input_fp32, kernel_size, stride, padding, dilation, ceil_mode);
  return at::quantize_per_tensor(
      result_fp32, input.q_scale(), input.q_zero_point(), input.scalar_type());
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("_adaptive_avg_pool2d", TORCH_FN(AdaptiveAvgPool2dQuantized));
  m.impl("quantized_max_pool2d", TORCH_FN(MaxPool2dQuantized));
}
} // namespace musa
} // namespace at