#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_QTENSOR_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_QTENSOR_H_
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/quantized/QTensorImpl.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/quantized/Quantizer.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace native {

// quantization functions, which take musa tensors in and quantize them
Tensor QuantizePerTensor(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    ScalarType dtype);

Tensor QuantizePerTensorTensorQParams(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    ScalarType dtype);

Tensor QuantizePerChannel(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType dtype);

Tensor QuantizePerTensorDynamic(
    const Tensor& self,
    ScalarType dtype,
    bool reduce_range);

// quantization attributes functions
// a quantized tensor (Qtensor) should contains below methods
double QScaleQuant(const Tensor& self);

int64_t QZeroPointQuant(const Tensor& self);

Tensor QPerChannelScales(const Tensor& self);

Tensor QPerChannelZeroPoints(const Tensor& self);

int64_t QPerChannelAxis(const Tensor& self);

QScheme QSchemeQuant(const Tensor& self);

Tensor DequantizeQuantized(const Tensor& self);

Tensor& QTensorCopy(Tensor& self, const Tensor& src);

} // namespace native
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_QTENSOR_H_
