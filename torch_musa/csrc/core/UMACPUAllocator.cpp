#include "UMACPUAllocator.h"
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>

#include <cstring>

#include <musa_runtime_api.h>
#include "torch_musa/csrc/core/MUSAException.h"
namespace c10 {
DataPtr UMACPUAllocator::allocate(size_t size) {
  void* ptr = nullptr;
  C10_MUSA_CHECK(musaMallocManaged(&ptr, size, musaMemAttachGlobal));
  return DataPtr(ptr, ptr, &raw_delete, c10::Device(c10::DeviceType::CPU));
}

void UMACPUAllocator::raw_delete(void* ctx) {
  if (ctx) {
    C10_MUSA_CHECK(musaFree(ctx));
  }
}

DeleterFnPtr UMACPUAllocator::raw_deleter() const {
  return &raw_delete;
}

void UMACPUAllocator::copy_data(void* dest, const void* src, std::size_t count)
    const {
  std::memcpy(dest, src, count);
}

static UMACPUAllocator uma_cpu_alloc;

void UMACPUAllocatorContext::set_allocator() {
  prev_allocator_ptr_ = c10::GetAllocator(c10::DeviceType::CPU);
  c10::SetAllocator(c10::DeviceType::CPU, &uma_cpu_alloc);
}

void UMACPUAllocatorContext::reset_allocator() {
  c10::SetAllocator(c10::DeviceType::CPU, prev_allocator_ptr_);
}
} // namespace c10
