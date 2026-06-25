#ifndef TORCH_MUSA_CSRC_CORE_CACHINGHOSTALLOCATOR_H_
#define TORCH_MUSA_CSRC_CORE_CACHINGHOSTALLOCATOR_H_

#include <ATen/core/CachingHostAllocator.h>
#include <c10/util/Deprecated.h>

#include "torch_musa/csrc/core/MUSAStream.h"

namespace at::musa {

//
// A caching allocator for MUSA host allocations (pinned memory).
//
// This provides a drop-in replacement for THMusaHostAllocator, which re-uses
// freed pinned (page-locked) memory allocations. This avoids device
// synchronizations due to cudaFreeHost calls.
//
// To ensure correct behavior, THCCachingHostAllocator_recordEvent must be
// called anytime a pointer from this allocator is used in a musaMemcpyAsync
// call between host and device, and passed the corresponding context from the
// allocation. This is currently invoked by at::native::copy_kernel_musa.
//
C10_DEPRECATED_MESSAGE(
    "at::musa::getCachingHostAllocator() is deprecated. Please use at::getHostAllocator(at::kMUSA) instead.")
inline at::HostAllocator* getCachingHostAllocator() {
  return at::getHostAllocator(at::kMUSA);
}

// Records an event in the specified stream. The allocation corresponding to the
// input `ptr`/`ctx` will not be re-used until the event has occurred.
C10_DEPRECATED_MESSAGE(
    "at::musa::CachingHostAllocator_recordEvent(...) is deprecated. Please use at::getHostAllocator(at::kMUSA)->record_event(...) instead.")
inline bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::musa::MUSAStream stream) {
  return getHostAllocator(at::kMUSA)->record_event(ptr, ctx, stream.unwrap());
}

// Releases cached pinned memory allocations via cudaHostFree
C10_DEPRECATED_MESSAGE(
    "at::musa::CachingHostAllocator_emptyCache() is deprecated. Please use at::getHostAllocator(at::kMUSA)->empty_cache() instead.")
inline void CachingHostAllocator_emptyCache() {
  getHostAllocator(at::kMUSA)->empty_cache();
}

C10_DEPRECATED_MESSAGE(
    "at::musa::HostAlloc(...) is deprecated. Please use at::getHostAllocator(at::kMUSA)->allocate(...) instead.")
inline at::DataPtr HostAlloc(size_t size) {
  return getHostAllocator(at::kMUSA)->allocate(size);
}

C10_DEPRECATED_MESSAGE(
    "at::musa::CachingHostAllocator_getStats() is deprecated. Please use at::getHostAllocator(at::kMUSA)->get_stats() instead.")
inline at::HostStats CachingHostAllocator_getStats() {
  return getHostAllocator(at::kMUSA)->get_stats();
}

C10_DEPRECATED_MESSAGE(
    "at::musa::CachingHostAllocator_resetAccumulatedStats() is deprecated. Please use at::getHostAllocator(at::kMUSA)->reset_accumulated_stats() instead.")
inline void CachingHostAllocator_resetAccumulatedStats() {
  getHostAllocator(at::kMUSA)->reset_accumulated_stats();
}

C10_DEPRECATED_MESSAGE(
    "at::musa::CachingHostAllocator_resetPeakStats() is deprecated. Please use at::getHostAllocator(at::kMUSA)->reset_peak_stats() instead.")
inline void CachingHostAllocator_resetPeakStats() {
  getHostAllocator(at::kMUSA)->reset_peak_stats();
}

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_CORE_CACHINGHOSTALLOCATOR_H_
