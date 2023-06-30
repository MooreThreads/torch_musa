#include "torch_musa/csrc/core/CachingHostAllocator.h"

namespace at {
namespace musa {

c10::Allocator* getCachingHostAllocator() {
  C10_THROW_ERROR(
      NotImplementedError,
      "getCachingHostAllocator in torch_musa is not supported now!");
  return nullptr;
}

bool CachingHostAllocator_recordEvent(void* ptr, void* ctx, MUSAStream stream) {
  C10_THROW_ERROR(
      NotImplementedError,
      "CachingHostAllocator_recordEvent in torch_musa is not supported now!");
  return false;
}

void CachingHostAllocator_emptyCache() {
  C10_THROW_ERROR(
      NotImplementedError,
      "CachingHostAllocator_emptyCache in torch_musa is not supported now!");
}

} // namespace musa
} // namespace at
