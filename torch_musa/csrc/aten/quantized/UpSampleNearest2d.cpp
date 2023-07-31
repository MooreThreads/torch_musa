#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/UpSample.h>

#include <torch/library.h>

#include "torch_musa/csrc/aten/quantized/QTensor.h"

namespace at {
namespace musa {
Tensor UpsampleNearest2dQuantized(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scale_h,
    c10::optional<double> scale_w) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, input, "UpsampleNearest2dQuantized", "input");

  const auto input_fp32 = input.dequantize();
  const auto dtype = input.scalar_type();
  const auto qscheme = input.qscheme();

  Tensor output;
  auto output_fp32 =
      at::upsample_nearest2d(input_fp32, output_size, scale_h, scale_w);

  if (qscheme == kPerTensorAffine) {
    output = at::musa::QuantizePerTensor(
        output_fp32, input.q_scale(), input.q_zero_point(), dtype);
  } else if (
      qscheme == kPerChannelAffine ||
      qscheme == kPerChannelAffineFloatQParams) {
    output = at::musa::QuantizePerChannel(
        output_fp32,
        input.q_per_channel_scales(),
        input.q_per_channel_zero_points(),
        input.q_per_channel_axis(),
        dtype);
  } else {
    TORCH_CHECK(
        false,
        "QScheme not supported by upsample_nearest2d:",
        toString(qscheme));
  }
  return output;
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("upsample_nearest2d", TORCH_FN(UpsampleNearest2dQuantized));
}

} // namespace musa
} // namespace at
