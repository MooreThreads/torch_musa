#include "torch_musa/csrc/core/MUSAUnifiedAllocator.h"
#include <c10/core/CPUAllocator.h>
#include <c10/util/Logging.h>
#include <musa_runtime_api.h>
#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAPluggableAllocator.h"
namespace c10 {
void* musa_malloc_managed(size_t size, int device, musaStream_t stream) {
  void* ptr = nullptr;
  C10_MUSA_CHECK(musaMallocManaged(&ptr, size, musaMemAttachGlobal));
  return ptr;
}

void musa_free(void* ptr, size_t size, int device, musaStream_t stream) {
  C10_MUSA_CHECK(musaFree(ptr));
}

std::shared_ptr<c10::musa::MUSACachingAllocator::MUSAAllocator>
get_pluggable_allocator() {
  return torch::musa::MUSAPluggableAllocator::createCustomAllocator(
      musa_malloc_managed, musa_free);
}

void register_unified_allocator() {
  static auto allocator =
      torch::musa::MUSAPluggableAllocator::createCustomAllocator(
          musa_malloc_managed, musa_free);
  c10::musa::MUSACachingAllocator::allocator.store(allocator.get());
  c10::SetAllocator(c10::kPrivateUse1, allocator.get());
}
} // namespace c10
