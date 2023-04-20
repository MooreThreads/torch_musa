#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_MTGPUTENSORFACTORY_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_MTGPUTENSORFACTORY_H_

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

Tensor empty_mtgpu(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt);

bool IsContiguous(
    const Tensor& self,
    MemoryFormat memory_format = MemoryFormat::Contiguous);
// Contiguous create new tensor when self tensor's storage_offset > 0 and
// not contiguous
Tensor Contiguous(
    const Tensor& self,
    MemoryFormat memory_format = MemoryFormat::Contiguous);

} // namespace musa
} // namespace native
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_MTGPUTENSORFACTORY_H_
