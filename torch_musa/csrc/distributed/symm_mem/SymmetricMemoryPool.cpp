#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <memory>

#include "torch_musa/csrc/core/MUSAPluggableAllocator.h"

namespace {

// Alloc functor for MemPool
void* symm_mem_alloc(size_t size, int device, void* stream) {
  static auto allocator =
      c10d::symmetric_memory::get_allocator(c10::DeviceType::PrivateUse1);
  // NOTE: for MUSA backend, we should pass group_name when rendezvous if the
  // symmetric memory was allocated from memory pool, because we can not access
  // the group_name under the alloc functor.
  return allocator->alloc(size, device, /*group_name=*/std::nullopt);
}

// Free functor for MemPool
void symm_mem_free(void* ptr, size_t size, int device, void* stream) {
  static auto allocator =
      c10d::symmetric_memory::get_allocator(c10::DeviceType::PrivateUse1);
  allocator->free(ptr);
}

} // namespace

struct RegisterMUSASymmetricMemoryPoolAllocator {
  RegisterMUSASymmetricMemoryPoolAllocator() {
    std::shared_ptr<c10::musa::MUSACachingAllocator::MUSAAllocator> allocator =
        torch::musa::MUSAPluggableAllocator::createCustomAllocator(
            symm_mem_alloc, symm_mem_free);
    c10d::symmetric_memory::register_mempool_allocator(
        c10::DeviceType::PrivateUse1, allocator);
  }
};

static RegisterMUSASymmetricMemoryPoolAllocator
    register_musa_symm_mem_allocator_;
