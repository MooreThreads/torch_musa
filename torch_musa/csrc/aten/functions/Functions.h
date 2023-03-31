#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_MTFUNCTIONS_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_MTFUNCTIONS_H_

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/native/TensorFactories.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace musa {

Tensor ClampMinExport(const Tensor& self, const Scalar& min);

Tensor DivTensorExport(const Tensor& self, const Tensor& other);

Tensor MeanExport(const Tensor& self, c10::optional<ScalarType> dtype);

Tensor SumExport(const Tensor& self, c10::optional<ScalarType> dtype);

} // namespace musa
} // namespace native
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_MTFUNCTIONS_H_
