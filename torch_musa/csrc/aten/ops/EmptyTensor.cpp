#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <c10/core/CPUAllocator.h>
#include <torch/library.h>

#include "torch_musa/csrc/core/CachingHostAllocator.h"
#include "torch_musa/csrc/core/PinnedMemoryAllocator.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace musa {

static c10::Allocator* GetCPUAllocatorMaybePinned(bool pin_memory) {
  if (pin_memory) {
    return getPinnedMemoryAllocator();
  }
  return c10::GetCPUAllocator();
}

at::TensorBase empty_cpu(
    c10::IntArrayRef size,
    at::ScalarType dtype,
    bool pin_memory,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return at::detail::empty_generic(
      size, allocator, cpu_ks, dtype, memory_format_opt);
}

at::TensorBase empty_cpu(
    c10::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_or_default(device_opt).type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      layout_or_default(layout_opt) == Layout::Strided);

  auto pin_memory = pinned_memory_or_default(pin_memory_opt);
  auto dtype = dtype_or_default(dtype_opt);
  return empty_cpu(size, dtype, pin_memory, memory_format_opt);
}

at::TensorBase empty_cpu(
    c10::IntArrayRef size,
    const at::TensorOptions& options) {
  return at::detail::empty_cpu(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  return empty_cpu(
      size,
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt,
      memory_format_opt);
}

OVERRIDE_CPU_OPERATOR_REGISTER_WITHOUT_WARNING(
    "empty.memory_format",
    empty_memory_format)

// TODO(mt-ai): Override empty_strided if needed. However, it seems not working
// well right now.

} // namespace musa
} // namespace at
