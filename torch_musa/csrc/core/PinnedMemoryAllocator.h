#ifndef TORCH_MUSA_CSRC_CORE_PINNEDMEMORYALLOCATOR_H_
#define TORCH_MUSA_CSRC_CORE_PINNEDMEMORYALLOCATOR_H_

#include <c10/core/Allocator.h>

#include "torch_musa/csrc/core/CachingHostAllocator.h"

namespace at {
namespace musa {

inline at::Allocator* getPinnedMemoryAllocator() {
  C10_THROW_ERROR(
      NotImplementedError,
      "PinnedMemoryAllocator in torch_musa is not supported now!");
  return nullptr;
}
} // namespace musa
} // namespace at

#endif // TORCH_MUSA_CSRC_CORE_PINNEDMEMORYALLOCATOR_H_
