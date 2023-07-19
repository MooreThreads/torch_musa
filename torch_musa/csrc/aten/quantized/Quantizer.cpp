#include <ATen/NativeFunctions.h>
#include <ATen/ceil_div.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/quantized/QTensorImpl.h>
#include <c10/core/CPUAllocator.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/quantized/Quantizer.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Allocator.h"

namespace at {

namespace {

void CheckPerChannelParamDims(const Tensor& scales, const Tensor& zero_points) {
  TORCH_CHECK(scales.dim() == 1, "scale tensor must have dimension 1");
  TORCH_CHECK(
      zero_points.dim() == 1, "zero_points tensor must have dimension 1");
  TORCH_CHECK(
      scales.numel() == zero_points.numel(),
      "number of elements in scales and zero_points must match, while getting ",
      scales.numel(),
      " vs ",
      zero_points.numel());
}

} // anonymous namespace

// Note: this is not a native function as Quantizer is not exposed to python yet
QuantizerPtr TensorBase::quantizer() const {
  return GetQTensorImpl(*this)->quantizer();
}

QuantizerPtr MakePerTensorAffineQuantizer(
    double scale,
    int64_t zero_point,
    ScalarType scalar_type) {
  return c10::make_intrusive<MusaPerTensorAffineQuantizer>(
      scalar_type, scale, zero_point);
}

QuantizerPtr MakePerChannelAffineQuantizer(
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType scalar_type) {
  CheckPerChannelParamDims(scales, zero_points);
  TORCH_CHECK(
      isFloatingType(scales.scalar_type()),
      "scale tensor must be floating point");

  if (isFloatingType(zero_points.scalar_type())) {
    Tensor scales_float = scales.to(kFloat).contiguous();
    Tensor zero_points_float = zero_points.to(kFloat).contiguous();
    return c10::make_intrusive<MusaPerChannelAffineFloatQParamsQuantizer>(
        scalar_type, scales_float, zero_points_float, axis);
  } else {
    Tensor scales_double = scales.to(kDouble).contiguous();
    Tensor zero_points_int64 = zero_points.to(kLong).contiguous();
    return c10::make_intrusive<MusaPerChannelAffineQuantizer>(
        scalar_type, scales_double, zero_points_int64, axis);
  }
}

int64_t GetSubByteTensorSize(
    IntArrayRef sizes,
    size_t dtype_itemsize,
    at::ScalarType t) {
  int64_t element_per_byte = 0;
  switch (t) {
    case at::ScalarType::QUInt4x2:
      element_per_byte = 2;
      break;
    case at::ScalarType::QUInt2x4:
      element_per_byte = 4;
      break;
    default:
      element_per_byte = 1;
  }
  // zero dim tensor
  if (sizes.empty()) {
    return c10::multiply_integers(sizes) * dtype_itemsize;
  }
  // Consider most inner dim as cols
  int64_t cols = sizes.at(sizes.size() - 1);
  int64_t bytes_per_row = cols * dtype_itemsize;
  // align qtensor most inner dim, compute ceil (bytes_per_row /
  // element_per_byte)
  return c10::multiply_integers(IntArrayRef(sizes.data(), sizes.size() - 1)) *
      at::ceil_div(bytes_per_row, element_per_byte);
}

inline Tensor NewQTensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer) {
  auto memory_format =
      options.memory_format_opt().value_or(MemoryFormat::Contiguous);
  auto device = options.device();
  at::Allocator* allocator = nullptr;
  if (device.is_privateuseone()) {
    allocator = c10::musa::MUSACachingAllocator::get();
  } else if (device.is_cpu()) {
    allocator = at::getCPUAllocator();
  } else {
    TORCH_INTERNAL_ASSERT(0, "unrecognized device for new_qtensor: ", device);
  }

  at::DispatchKey tensorDispatchKey = options.computeDispatchKey();
  native::check_size_nonnegative(sizes);
  auto dtype = options.dtype();
  TORCH_CHECK(
      isQIntType(typeMetaToScalarType(dtype)),
      "ScalarType ",
      typeMetaToScalarType(dtype),
      " is not supported in new_qtensor.");
  auto scalar_type = typeMetaToScalarType(dtype);
  int64_t size_bytes =
      GetSubByteTensorSize(sizes, dtype.itemsize(), scalar_type);

  auto storage = c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizable=*/true);
  auto tensor = detail::make_tensor<QTensorImpl>(
      storage, at::DispatchKeySet(tensorDispatchKey), dtype, quantizer);
  GetQTensorImpl(tensor)->set_sizes_contiguous(sizes);
  GetQTensorImpl(tensor)->empty_tensor_restride(memory_format);
  return tensor;
}

Tensor MusaPerTensorAffineQuantizer::quantize(const Tensor& rtensor) {
  TORCH_CHECK(
      rtensor.scalar_type() == kFloat,
      "Quantize only works on Float Tensor, got ",
      rtensor.scalar_type());
  // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
  // quantizer that can be reused, so I'm using intrusive_from_this here
  Tensor qtensor = NewQTensor(
      rtensor.sizes(),
      rtensor.options()
          .dtype(scalar_type_)
          .memory_format(rtensor.suggest_memory_format()),
      intrusive_from_this());

  auto rtensor_contig =
      rtensor.expect_contiguous(rtensor.suggest_memory_format());
  native::quantize_tensor_per_tensor_affine(
      *rtensor_contig, qtensor, scale_, zero_point_);
  return qtensor;
}

static void PerTensorAffineDequantizeImpl(
    Tensor& rtensor,
    const Tensor& qtensor,
    const double scale,
    const int64_t zero_point) {
  const auto qtensor_contig =
      qtensor.expect_contiguous(qtensor.suggest_memory_format());
  native::dequantize_tensor_per_tensor_affine(
      *qtensor_contig, rtensor, scale, zero_point);
}

Tensor& MusaPerTensorAffineQuantizer::dequantize_out(
    Tensor& rtensor,
    const Tensor& qtensor) {
  rtensor.resize_(qtensor.sizes());
  TORCH_CHECK(
      rtensor.is_contiguous(qtensor.suggest_memory_format()) &&
          rtensor.scalar_type() == kFloat,
      "Dequantize out should be a contiguous Float Tensor; instead got type ",
      rtensor.scalar_type(),
      ", and is_contiguous ",
      rtensor.is_contiguous(qtensor.suggest_memory_format()));
  PerTensorAffineDequantizeImpl(rtensor, qtensor, scale_, zero_point_);
  return rtensor;
}

Tensor MusaPerTensorAffineQuantizer::dequantize(const Tensor& qtensor) {
  Tensor rtensor = at::empty(
      qtensor.sizes(),
      qtensor.options()
          .dtype(at::kFloat)
          .memory_format(qtensor.suggest_memory_format()));
  PerTensorAffineDequantizeImpl(rtensor, qtensor, scale_, zero_point_);
  return rtensor;
}

Tensor MusaPerChannelAffineQuantizer::quantize(const Tensor& rtensor) {
  // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
  // quantizer that can be reused, so I'm using intrusive_from_this here
  Tensor qtensor = NewQTensor(
      rtensor.sizes(),
      rtensor.options()
          .dtype(scalar_type_)
          .memory_format(rtensor.suggest_memory_format()),
      intrusive_from_this());
  auto rtensor_contig =
      rtensor.expect_contiguous(rtensor.suggest_memory_format());
  native::quantize_tensor_per_channel_affine(
      *rtensor_contig, qtensor, scales_, zero_points_, axis_);
  return qtensor;
}

static void PerChannelAffineDequantizeImpl(
    Tensor& rtensor,
    const Tensor& qtensor,
    const Tensor& scale,
    const Tensor& zero_point,
    const int64_t axis) {
  const auto qtensor_contig =
      qtensor.expect_contiguous(qtensor.suggest_memory_format());
  native::dequantize_tensor_per_channel_affine(
      *qtensor_contig, rtensor, scale, zero_point, axis);
}

Tensor MusaPerChannelAffineQuantizer::dequantize(const Tensor& qtensor) {
  Tensor rtensor = at::empty(
      qtensor.sizes(),
      qtensor.options()
          .dtype(at::kFloat)
          .memory_format(qtensor.suggest_memory_format()));
  PerChannelAffineDequantizeImpl(
      rtensor, qtensor, scales_, zero_points_, axis_);
  return rtensor;
}

Tensor& MusaPerChannelAffineQuantizer::dequantize_out(
    Tensor& rtensor,
    const Tensor& qtensor) {
  rtensor.resize_(qtensor.sizes());
  TORCH_CHECK(
      rtensor.is_contiguous(qtensor.suggest_memory_format()) &&
          rtensor.scalar_type() == kFloat,
      "Dequantize out should be a contiguous Float Tensor; instead got type ",
      rtensor.scalar_type(),
      ", and is_contiguous ",
      rtensor.is_contiguous(qtensor.suggest_memory_format()));
  PerChannelAffineDequantizeImpl(
      rtensor, qtensor, scales_, zero_points_, axis_);
  return rtensor;
}

Tensor MusaPerChannelAffineFloatQParamsQuantizer::quantize(
    const Tensor& rtensor) {
  TORCH_CHECK(
      rtensor.scalar_type() == kFloat,
      "Quantize only works on Float Tensor, got ",
      rtensor.scalar_type());
  Tensor qtensor = NewQTensor(
      rtensor.sizes(),
      rtensor.options().dtype(scalar_type_),
      intrusive_from_this());
  auto rtensor_contig = rtensor.expect_contiguous();
  native::quantize_tensor_per_channel_float_qparams(
      *rtensor_contig, qtensor, scales_, zero_points_, axis_);
  return qtensor;
}

static void PerChannelAffineFloatQParamsDequantizeImpl(
    Tensor& rtensor,
    const Tensor& qtensor,
    const Tensor& scale,
    const Tensor& zero_point,
    const int64_t axis) {
  const auto qtensor_contig =
      qtensor.expect_contiguous(qtensor.suggest_memory_format());
  native::dequantize_tensor_per_channel_float_qparams(
      *qtensor_contig, rtensor, scale, zero_point, axis);
}

Tensor MusaPerChannelAffineFloatQParamsQuantizer::dequantize(
    const Tensor& qtensor) {
  Tensor rtensor =
      at::empty(qtensor.sizes(), qtensor.options().dtype(at::kFloat));
  PerChannelAffineFloatQParamsDequantizeImpl(
      rtensor, qtensor, scales_, zero_points_, axis_);
  return rtensor;
}

Quantizer::~Quantizer() = default;

Tensor& MusaPerChannelAffineFloatQParamsQuantizer::dequantize_out(
    Tensor& rtensor,
    const Tensor& qtensor) {
  rtensor.resize_(qtensor.sizes());
  TORCH_CHECK(
      rtensor.is_contiguous(qtensor.suggest_memory_format()) &&
          rtensor.scalar_type() == kFloat,
      "Dequantize out should be a contiguous Float Tensor; instead got type ",
      rtensor.scalar_type(),
      ", and is_contiguous ",
      rtensor.is_contiguous(qtensor.suggest_memory_format()));
  PerChannelAffineFloatQParamsDequantizeImpl(
      rtensor, qtensor, scales_, zero_points_, axis_);
  return rtensor;
}

Tensor MusaUnknownQuantizer::quantize(const Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(false, "cannot call quantize on UnknownQuantizer");
}
Tensor MusaUnknownQuantizer::dequantize(const Tensor& qtensor) {
  TORCH_INTERNAL_ASSERT(false, "cannot call dequantize on UnknownQuantizer");
}
Tensor& MusaUnknownQuantizer::dequantize_out(
    Tensor& rtensor,
    const Tensor& qtensor) {
  TORCH_INTERNAL_ASSERT(
      false, "cannot call dequantize_out on UnknownQuantizer");
}
QScheme MusaUnknownQuantizer::qscheme() const {
  TORCH_INTERNAL_ASSERT(false, "cannot call qscheme on UnknownQuantizer");
}
bool MusaUnknownQuantizer::equalTo(QuantizerPtr other) const {
  TORCH_INTERNAL_ASSERT(false, "cannot call equalTo on UnknownQuantizer");
}

QuantizerPtr MakeUnknownQuantizer(ScalarType scalar_type) {
  return c10::make_intrusive<MusaUnknownQuantizer>(scalar_type);
}

QTensorImpl* GetQTensorImpl(const TensorBase& self) {
  TORCH_CHECK(
      !self.requires_grad(), "quantized tensors do not support autograd");
  TORCH_INTERNAL_ASSERT(
      self.is_quantized(), "get_qtensorimpl: not a quantized tensor");
  return static_cast<QTensorImpl*>(self.unsafeGetTensorImpl());
}

} // namespace at
