#ifndef MUSA_MANAGED_ALLOCATOR_H
#define MUSA_MANAGED_ALLOCATOR_H

#include <musa_runtime_api.h>
#include <cstddef> // for size_t
#include <memory>
#include "torch_musa/csrc/core/MUSACachingAllocator.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace c10 {

void* musa_malloc_managed(size_t size, int device, musaStream_t stream);

void musa_free(void* ptr, size_t size, int device, musaStream_t stream);

std::shared_ptr<c10::musa::MUSACachingAllocator::MUSAAllocator>
get_pluggable_allocator();

void register_unified_allocator();
} // namespace c10

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MUSA_MANAGED_ALLOCATOR_H
