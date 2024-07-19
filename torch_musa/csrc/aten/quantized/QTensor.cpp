#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/quantized/QTensorImpl.h>
#include <torch/library.h>

#include <ATen/core/op_registration/adaption.h>
#include "torch_musa/csrc/aten/quantized/Quantizer.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

inline std::tuple<int, int> QValueRangeHelper(c10::ScalarType dtype) {
  if (dtype == c10::ScalarType::QUInt8) {
    return {0, 255};
  } else if (dtype == c10::ScalarType::QInt32) {
    return {-128, 127}; // use qint32 to simulate qint8
  } else {
    TORCH_CHECK(
        false, "Currently mudnn only supports QUInt8 and QINT32, got: ", dtype);
  }
}

namespace at {
namespace musa {

Tensor QuantizePerTensor(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    ScalarType dtype) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "QuantizePerTensor", "self");
  const OptionalDeviceGuard device_guard(device_of(self));
  auto quantizer = at::MakePerTensorAffineQuantizer(scale, zero_point, dtype);
  return quantizer->quantize(self);
}

Tensor QuantizePerTensorTensorQParams(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    ScalarType dtype) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "QuantizePerTensorTensorQParams", "self");
  c10::impl::check_and_update_common_device(
      common_device, scale, "QuantizePerTensorTensorQParams", "scale");
  c10::impl::check_and_update_common_device(
      common_device,
      zero_point,
      "QuantizePerTensorTensorQParams",
      "zero_point");
  const OptionalDeviceGuard device_guard(device_of(self));
  auto quantizer = at::MakePerTensorAffineQuantizer(
      scale.item().toDouble(), zero_point.item().toLong(), dtype);
  return quantizer->quantize(self);
}

Tensor QuantizePerChannel(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType dtype) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "QuantizePerChannel", "self");
  c10::impl::check_and_update_common_device(
      common_device, scales, "QuantizePerChannel", "scales");
  c10::impl::check_and_update_common_device(
      common_device, zero_points, "QuantizePerChannel", "zero_points");
  const OptionalDeviceGuard device_guard(device_of(self));
  auto quantizer =
      at::MakePerChannelAffineQuantizer(scales, zero_points, axis, dtype);
  return quantizer->quantize(self);
}

Tensor QuantizePerTensorDynamic(
    const Tensor& self,
    ScalarType dtype,
    bool reduce_range) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "QuantizePerTensorDynamic", "self");
  const OptionalDeviceGuard device_guard(device_of(self));
  TORCH_CHECK(
      (dtype == ScalarType::QInt8 || dtype == ScalarType::QUInt8 ||
       dtype == ScalarType::Half),
      "dtype",
      dtype,
      "not supported");
  auto input_config = self.contiguous();
  if (dtype == ScalarType::Half) {
    return input_config.to(ScalarType::Half);
  }
  float x_min = input_config.min().item<float>();
  float x_max = input_config.max().item<float>();

  // QNNPACK doesn't support reduce_range argument, currently only FBGEMM
  // supports it
  if (reduce_range && at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    reduce_range = false;
  }

  auto [qmin, qmax] = QValueRangeHelper(dtype);

  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/qmin,
      /*qmax=*/qmax,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/reduce_range);

  return QuantizePerTensor(self, q_params.scale, q_params.zero_point, dtype);
}

double QScaleQuant(const Tensor& self) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "QScaleQuant", "self");
  const OptionalDeviceGuard device_guard(device_of(self));
  auto quantizer = at::GetQTensorImpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerTensorAffine);
  return static_cast<PerTensorAffineQuantizer*>(quantizer.get())->scale();
}

int64_t QZeroPointQuant(const Tensor& self) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "QZeroPointQuant", "self");
  const OptionalDeviceGuard device_guard(device_of(self));
  auto quantizer = at::GetQTensorImpl(self)->quantizer();
  TORCH_CHECK(quantizer->qscheme() == kPerTensorAffine);
  return static_cast<PerTensorAffineQuantizer*>(quantizer.get())->zero_point();
}

Tensor QPerChannelScales(const Tensor& self) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "QPerChannelScales", "self");
  const OptionalDeviceGuard device_guard(device_of(self));
  auto quantizer = at::GetQTensorImpl(self)->quantizer();
  TORCH_CHECK(
      quantizer->qscheme() == kPerChannelAffine ||
      quantizer->qscheme() == kPerChannelAffineFloatQParams);
  return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->scales();
}

Tensor QPerChannelZeroPoints(const Tensor& self) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "QPerChannelZeroPoints", "self");
  const OptionalDeviceGuard device_guard(device_of(self));
  auto quantizer = at::GetQTensorImpl(self)->quantizer();
  TORCH_CHECK(
      quantizer->qscheme() == kPerChannelAffine ||
      quantizer->qscheme() == kPerChannelAffineFloatQParams);
  return static_cast<PerChannelAffineQuantizer*>(quantizer.get())
      ->zero_points();
}

int64_t QPerChannelAxis(const Tensor& self) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "QPerChannelAxis", "self");
  const OptionalDeviceGuard device_guard(device_of(self));
  auto quantizer = at::GetQTensorImpl(self)->quantizer();
  TORCH_CHECK(
      quantizer->qscheme() == kPerChannelAffine ||
      quantizer->qscheme() == kPerChannelAffineFloatQParams);
  return static_cast<PerChannelAffineQuantizer*>(quantizer.get())->axis();
}

Tensor& SetStorageQuantized(
    Tensor& self,
    Storage storage,
    int64_t storage_offset,
    IntArrayRef sizes,
    IntArrayRef strides) {
  auto* self_ = self.unsafeGetTensorImpl();
  self_->set_storage_keep_dtype(std::move(storage));
  self_->set_storage_offset(storage_offset);
  self_->set_sizes_and_strides(sizes, strides);
  return self;
}

QScheme QSchemeQuant(const Tensor& self) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "QSchemeQuant", "self");
  const OptionalDeviceGuard device_guard(device_of(self));
  auto quantizer = at::GetQTensorImpl(self)->quantizer();
  return quantizer->qscheme();
}

Tensor DequantizeQuantized(const Tensor& self) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "DequantizeQuantized", "self");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::GetQTensorImpl(self)->quantizer()->dequantize(self);
}

Tensor QuantizedClone(
    const Tensor& self,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "QuantizedClone", "self");
  const OptionalDeviceGuard device_guard(device_of(self));
  auto memory_format =
      optional_memory_format.value_or(MemoryFormat::Contiguous);

  // TODO(fan.mo): To support all features of MemoryFormat::Preserve we need to
  // add _empty_affine_quantized_strided function and use it similarly to Tensor
  // clone(const Tensor& src, c10::optional<c10::MemoryFormat>
  // optional_memory_format) if (self.is_non_overlapping_and_dense()) ->
  // _empty_affine_quantized_strided
  if (memory_format == MemoryFormat::Preserve) {
    memory_format = self.suggest_memory_format();
  }

  Tensor dst;
  if (self.qscheme() == at::kPerTensorAffine) {
    dst = at::_empty_affine_quantized(
        self.sizes(),
        self.options().memory_format(memory_format),
        self.q_scale(),
        self.q_zero_point(),
        c10::nullopt);
  } else if (self.qscheme() == at::kPerChannelAffine) {
    dst = at::_empty_per_channel_affine_quantized(
        self.sizes(),
        self.q_per_channel_scales(),
        self.q_per_channel_zero_points(),
        self.q_per_channel_axis(),
        self.options().memory_format(memory_format),
        c10::nullopt);
  } else {
    TORCH_CHECK(
        false,
        "clone for quantized Tensor only works for \
      PerTensorAffine and PerChannelAffine qscheme right now");
  }

  at::native::copy_(dst, self, false);

  return dst;
}

Tensor& QTensorCopy(Tensor& self, const Tensor& src) {
  TORCH_CHECK(
      src.scalar_type() == at::kFloat,
      "Quantized copy only works with kFloat as source Tensor");
  TORCH_CHECK(
      (self.is_contiguous() && src.is_contiguous()) ||
          (self.is_contiguous(at::MemoryFormat::ChannelsLast) &&
           src.is_contiguous(at::MemoryFormat::ChannelsLast)),
      "Quantized copy only works with contiguous and NHWC Tensors");
  TORCH_CHECK(
      self.sizes().equals(src.sizes()),
      "Quantized copy only works with Tensor with the same shape");
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "Copy", [&]() {
    if (self.qscheme() == kPerChannelAffine ||
        self.qscheme() == kPerChannelAffineFloatQParams ||
        self.qscheme() == kPerChannelSymmetric) {
      at::native::quantize_tensor_per_channel_affine(
          src,
          self,
          self.q_per_channel_scales(),
          self.q_per_channel_zero_points(),
          self.q_per_channel_axis());
    } else {
      at::native::quantize_tensor_per_tensor_affine(
          src, self, self.q_scale(), self.q_zero_point());
    }
  });
  return self;
}

} // namespace musa
} // namespace at
