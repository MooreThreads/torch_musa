#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_TENSORFACTORY_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_TENSORFACTORY_H_

#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/native/TensorFactories.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>

#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
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

bool IsContiguous(
    const Tensor& self,
    MemoryFormat memory_format = MemoryFormat::Contiguous);
// Contiguous create new tensor when self tensor's storage_offset > 0 and
// not contiguous

Tensor Contiguous(
    const Tensor& self,
    Tensor& result,
    MemoryFormat memory_format = MemoryFormat::Contiguous);

Tensor Contiguous(
    const Tensor& self,
    MemoryFormat memory_format = MemoryFormat::Contiguous);

} // namespace musa
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_TENSORFACTORY_H_
