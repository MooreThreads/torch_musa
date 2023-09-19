#ifndef TORCH_MUSA_CSRC_CORE_CACHINGHOSTALLOCATOR_H_
#define TORCH_MUSA_CSRC_CORE_CACHINGHOSTALLOCATOR_H_

#include "torch_musa/csrc/core/Allocator.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace musa {

// A caching allocator for MUSA host allocations (pinned memory).
c10::Allocator* getCachingHostAllocator();

// Records an event in the specified stream. The allocation corresponding to the
// input `ptr`/`ctx` will not be re-used until the event has occurred.
bool CachingHostAllocator_recordEvent(void* ptr, void* ctx, MUSAStream stream);

// Releases cached pinned memory allocations via musaHostFree
void CachingHostAllocator_emptyCache();

inline at::DataPtr HostAlloc(size_t size) {
  return getCachingHostAllocator()->allocate(size);
}

} // namespace musa
} // namespace at
#endif // TORCH_MUSA_CSRC_CORE_CACHINGHOSTALLOCATOR_H_
