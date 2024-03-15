#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_TENSORFACTORY_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_TENSORFACTORY_H_

#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/native/TensorFactories.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
namespace detail {
Tensor empty_musa(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt);

TensorBase empty_musa(IntArrayRef size, const TensorOptions& options);

} // namespace detail

namespace musa {

Tensor empty_musa(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt);

Tensor empty_strided_musa(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt);

Tensor Contiguous(
    const Tensor& self,
    MemoryFormat memory_format = MemoryFormat::Contiguous);

Tensor ContiguousRef(
    const Tensor& self,
    Tensor& result,
    MemoryFormat memory_format = MemoryFormat::Contiguous);

} // namespace musa
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_TENSORFACTORY_H_
