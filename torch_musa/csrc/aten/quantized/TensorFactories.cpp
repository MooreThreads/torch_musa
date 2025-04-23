#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ceil_div.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/ops/_make_per_tensor_quantized_tensor.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
namespace musa {
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ empty ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// We explicitly pass in scale and zero_point because we don't have the infra
// ready to support quantizer in python frontend, once that is ready, we'll
// change to use quantizer
Tensor EmptyAffineQuantized(
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    double scale,
    int64_t zero_point,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  const DeviceGuard device_guard(device_or_default(device));
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  TORCH_CHECK(
      !(options_.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");
  auto options = options_.merge_memory_format(optional_memory_format);
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  return at::new_qtensor(
      size,
      options,
      at::make_per_tensor_affine_quantizer(
          scale, zero_point, typeMetaToScalarType(options.dtype())));
}

Tensor EmptyPerChannelAffineQuantized(
    IntArrayRef size,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  const DeviceGuard device_guard(device_or_default(device));
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  TORCH_CHECK(
      !(options_.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");
  auto options = options_.merge_memory_format(optional_memory_format);
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  QuantizerPtr quantizer = at::make_per_channel_affine_quantizer(
      scales.to(options.device()),
      zero_points.to(options.device()),
      axis,
      typeMetaToScalarType(options.dtype()));
  return at::new_qtensor(size, options, std::move(quantizer));
}

Tensor EmptyUnknownQuantized(
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  TORCH_CHECK(
      !(options_.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");
  auto options = options_.merge_memory_format(optional_memory_format);
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  QuantizerPtr quantizer =
      at::make_unknown_quantizer(typeMetaToScalarType(options.dtype()));
  return at::new_qtensor(size, options, std::move(quantizer));
}

// Create an empty quantized Tensor with size, based on the options
// and quantization parameters of the input quantized Tensor
Tensor EmptyQuantized(
    IntArrayRef size,
    const Tensor& qtensor,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format) {
  const DeviceGuard device_guard(device_or_default(device));
  TensorOptions specified_options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  TORCH_CHECK(
      !(specified_options.has_memory_format() && memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");

  TensorOptions options = qtensor.options()
                              .merge_in(specified_options)
                              .merge_memory_format(memory_format);

  Tensor output;
  if (qtensor.qscheme() == kPerTensorAffine) {
    output = at::_empty_affine_quantized(
        size, options, qtensor.q_scale(), qtensor.q_zero_point());
  } else if (
      qtensor.qscheme() == kPerChannelAffine ||
      qtensor.qscheme() == kPerChannelAffineFloatQParams) {
    output = at::_empty_per_channel_affine_quantized(
        size,
        qtensor.q_per_channel_scales(),
        qtensor.q_per_channel_zero_points(),
        qtensor.q_per_channel_axis(),
        options);
  } else {
    TORCH_CHECK(
        false,
        "QScheme not supported by empty_quantized:",
        toString(qtensor.qscheme()));
  }
  return output;
}

} // namespace musa
} // namespace at
