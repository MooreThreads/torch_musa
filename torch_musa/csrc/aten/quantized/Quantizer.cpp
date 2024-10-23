#include <ATen/NativeFunctions.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/quantized/QTensorImpl.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Allocator.h"
#include "torch_musa/csrc/core/MUSAHooksInterface.h"

namespace at {

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

} // namespace at
