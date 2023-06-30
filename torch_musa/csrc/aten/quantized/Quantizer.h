#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_QUANTIZER_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_QUANTIZER_H_
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>

#include <ATen/Tensor.h>
#include <ATen/TensorUtils.h>

#include <ATen/core/QuantizerBase.h>

#include <cmath>
#include <memory>
#include <utility>

namespace at {

/**
 * UnknownQuantizer is a placeholder quantizer for functions that implement
 * quantization in a two step process.  First a tensor is allocated but with
 * unknown quantizer, and then the quantization kernel decides what the final
 * quantizer will be.
 */
struct TORCH_API MusaUnknownQuantizer : public Quantizer {
  explicit MusaUnknownQuantizer(ScalarType scalar_type)
      : Quantizer(scalar_type) {}

  Tensor quantize(const Tensor& tensor) override;
  Tensor dequantize(const Tensor& qtensor) override;
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;
  QScheme qscheme() const override;
  bool equalTo(QuantizerPtr other) const override;
};

/**
 * PerTensorAffineQuantizer stores a scale and a zero_point, which is used for
 * all the values in the Tensor.
 */
struct TORCH_API MusaPerTensorAffineQuantizer : public AffineQuantizer {
  explicit MusaPerTensorAffineQuantizer(
      ScalarType scalar_type,
      double scale,
      int64_t zero_point)
      : AffineQuantizer(scalar_type), scale_(scale), zero_point_(zero_point) {}

  Tensor quantize(const Tensor& tensor) override;
  Tensor dequantize(const Tensor& qtensor) override;
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;

  QScheme qscheme() const override {
    return kPerTensorAffine;
  }

  double scale() const {
    return scale_;
  }

  int64_t zero_point() const {
    return zero_point_;
  }

  bool equalTo(QuantizerPtr other) const override {
    if (!other.get() || other->qscheme() != kPerTensorAffine) {
      return false;
    }
    auto* other_per_tensor_affine =
        static_cast<MusaPerTensorAffineQuantizer*>(other.get());
    return scalar_type() == other_per_tensor_affine->scalar_type() &&
        scale() == other_per_tensor_affine->scale() &&
        zero_point() == other_per_tensor_affine->zero_point();
  }

 private:
  const double scale_;
  // We use int64_t for consistency with Python
  const int64_t zero_point_;
};

/**
 * PerChannelAffineQuantizer is the same as PerTensorAffineQuantizer
 * except that we have an independent scale and zero_point parameter
 * for each channel.
 *
 * Also note that per channel quantization is mostly applied to output channels
 * of weights since per-input channel of weight quantization or per-channel
 * quantization for activations can't be efficiently supported in most of
 * processors since it requires each multiplication result within a single
 * dot-product to have a different scale.
 */
struct TORCH_API MusaPerChannelAffineQuantizer : public AffineQuantizer {
  explicit MusaPerChannelAffineQuantizer(
      ScalarType scalar_type,
      Tensor scales,
      Tensor zero_points,
      int64_t axis)
      : AffineQuantizer(scalar_type),
        scales_(std::move(scales)),
        zero_points_(std::move(zero_points)),
        axis_(axis) {}

  QScheme qscheme() const override {
    return kPerChannelAffine;
  }

  Tensor scales() const {
    return scales_;
  }

  Tensor zero_points() const {
    return zero_points_;
  }

  int64_t axis() const {
    return axis_;
  }

  Tensor quantize(const Tensor& tensor) override;
  Tensor dequantize(const Tensor& qtensor) override;
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;

  bool equalTo(QuantizerPtr other) const override {
    if (!other.get() || other->qscheme() != kPerChannelAffine) {
      return false;
    }
    auto* other_per_channel_affine =
        static_cast<MusaPerChannelAffineQuantizer*>(other.get());
    return scalar_type() == other_per_channel_affine->scalar_type() &&
        scales().equal(other_per_channel_affine->scales()) &&
        zero_points().equal(other_per_channel_affine->zero_points()) &&
        axis() == other_per_channel_affine->axis();
  }

 protected:
  Tensor scales_;
  Tensor zero_points_;
  const int64_t axis_;
};

/**
 * PerChannelAffineFloatQParamsQuantizer is the same as
 * PerChannelAffineQuantizer except that it expects both scale and zero point to
 * be floating point values.
 *
 * This quantizer uses the kPerChannelAffineFloatQParams qscheme which is a
 * variant of kPerChannelAffine.
 *
 * The quantize equation in this case looks like -
 * Xq = (Xf - zero_point) * inv_scale, where inv_scale = 1.0/scale
 *
 * Note: Usage of floating point zero point is useful in cases where 0 doesn't
 * need to be exactly represented in the quantized space. We can get additional
 * precision by using floating point values for zero point.
 */
struct TORCH_API MusaPerChannelAffineFloatQParamsQuantizer
    : public PerChannelAffineQuantizer {
  explicit MusaPerChannelAffineFloatQParamsQuantizer(
      ScalarType scalar_type,
      Tensor scales,
      Tensor zero_points,
      int64_t axis)
      : PerChannelAffineQuantizer(scalar_type, scales, zero_points, axis) {}

  QScheme qscheme() const override {
    return kPerChannelAffineFloatQParams;
  }

  Tensor quantize(const Tensor& tensor) override;
  Tensor dequantize(const Tensor& qtensor) override;
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;

  bool equalTo(QuantizerPtr other) const override {
    if (!other.get() || other->qscheme() != kPerChannelAffineFloatQParams) {
      return false;
    }
    auto* other_per_channel_float_qparams =
        static_cast<MusaPerChannelAffineFloatQParamsQuantizer*>(other.get());
    return scalar_type() == other_per_channel_float_qparams->scalar_type() &&
        scales().equal(other_per_channel_float_qparams->scales()) &&
        zero_points().equal(other_per_channel_float_qparams->zero_points()) &&
        axis() == other_per_channel_float_qparams->axis();
  }
};

// double and int64_t are because of the native function API, we only have these
// argument types right now in native functions
TORCH_API QuantizerPtr MakePerTensorAffineQuantizer(
    double scale,
    int64_t zero_point,
    ScalarType scalar_type);

TORCH_API QuantizerPtr MakePerChannelAffineQuantizer(
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType scalar_type);

TORCH_API QuantizerPtr MakeUnknownQuantizer(ScalarType scalar_type);

// Create a Quantized Tensor given arguments for normal Tensor and a quantizer
TORCH_API Tensor NewQTensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer);

} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_QUANTIZER_H_
