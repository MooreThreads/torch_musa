#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/ops/_make_per_tensor_quantized_tensor.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/quantized/Quantizer.h"
#include "torch_musa/csrc/aten/quantized/TensorFactories.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace musa {

Tensor MakePerTensorQuantizedTensor(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::make_per_tensor_quantized_tensor_cuda(
      self, scale, zero_point);
}

Tensor MakePerChannelQuantizedTensor(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "MakePerChannelQuantizedTensor", "self");
  c10::impl::check_and_update_common_device(
      common_device, scales, "MakePerChannelQuantizedTensor", "scale");
  c10::impl::check_and_update_common_device(
      common_device,
      zero_points,
      "MakePerChannelQuantizedTensor",
      "zero_point");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::make_per_channel_quantized_tensor_cuda(
      self, scales, zero_points, axis);
}

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
  return NewQTensor(
      size,
      options,
      MakePerTensorAffineQuantizer(
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
  QuantizerPtr quantizer = MakePerChannelAffineQuantizer(
      scales.to(options.device()),
      zero_points.to(options.device()),
      axis,
      typeMetaToScalarType(options.dtype()));
  return NewQTensor(size, options, std::move(quantizer));
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
      MakeUnknownQuantizer(typeMetaToScalarType(options.dtype()));
  return NewQTensor(size, options, std::move(quantizer));
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
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, qtensor, "EmptyQuantized", "qtensor");
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

// Basically copied from pytorch official
Tensor EmptyLikeQuantized(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  // DeviceGuard omitted
  return at::native::empty_like_quantized(
      self, dtype, layout, device, pin_memory, optional_memory_format);
}

// Basically copied from pytorch official
Tensor EmptyStridedUnknownQuantized(
    IntArrayRef size,
    IntArrayRef strided,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  const DeviceGuard device_guard(device_or_default(device));
  return at::native::empty_strided_unknown_quantized(
      size, strided, dtype, layout, device, pin_memory);
}

Tensor AsStridedQTensorImpl(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    optional<int64_t> storage_offset_) {
  // DeviceGuard omitted
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto quantizer = at::GetQTensorImpl(self)->quantizer();
  TORCH_CHECK(
      quantizer->qscheme() == QScheme::PER_TENSOR_AFFINE,
      "Setting strides is possible only on uniformly quantized tensor");
  auto result = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW,
      Storage(self.storage()),
      self.key_set(),
      self.dtype(),
      quantizer);
  at::native::setStrided(result, size, stride, storage_offset);
  return result;
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(EmptyUnknownQuantized));
  m.impl("empty_quantized", TORCH_FN(EmptyQuantized));
  m.impl("_empty_affine_quantized", TORCH_FN(EmptyAffineQuantized));
  m.impl(
      "_empty_per_channel_affine_quantized",
      TORCH_FN(EmptyPerChannelAffineQuantized));
  m.impl("empty_like", TORCH_FN(EmptyLikeQuantized));
  m.impl("empty_strided", TORCH_FN(EmptyStridedUnknownQuantized));

  m.impl("as_strided", TORCH_FN(AsStridedQTensorImpl));

  m.impl("view", TORCH_FN(at::native::view));

  m.impl("_reshape_alias", TORCH_FN(at::native::_reshape_alias));
  m.impl("unfold", TORCH_FN(at::native::unfold));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_make_per_tensor_quantized_tensor", &MakePerTensorQuantizedTensor);
  m.impl("_make_per_channel_quantized_tensor", &MakePerChannelQuantizedTensor);
  m.impl("_empty_affine_quantized", TORCH_FN(EmptyAffineQuantized));
}
} // namespace musa
} // namespace at
