#ifndef TORCH_MUSA_CSRC_CORE_PINNEDMEMORYALLOCATOR_H_
#define TORCH_MUSA_CSRC_CORE_PINNEDMEMORYALLOCATOR_H_

#include "torch_musa/csrc/core/CachingHostAllocator.h"

namespace at::musa {

inline at::Allocator* getPinnedMemoryAllocator() {
  return getCachingHostAllocator();
}

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_CORE_PINNEDMEMORYALLOCATOR_H_
