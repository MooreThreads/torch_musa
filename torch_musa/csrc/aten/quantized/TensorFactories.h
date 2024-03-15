#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_TENSOR_FACTORIES_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_TENSOR_FACTORIES_H_
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorFactories.h>
#include <torch/library.h>

namespace at {
namespace musa {

Tensor MakePerTensorQuantizedTensor(
    const Tensor& self,
    double scale,
    int64_t zero_point);

Tensor MakePerChannelQuantizedTensor(
    const Tensor& self,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

} // namespace musa
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_TENSOR_FACTORIES_H_
