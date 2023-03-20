#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <list>
#pragma GCC diagnostic pop

#include <mudnn.h>

namespace musa {
class AutoGrowthBestFitAllocator {
 public:
  static AutoGrowthBestFitAllocator* get_allocator() {
    static AutoGrowthBestFitAllocator g_alloctator;
    return &g_alloctator;
  }

  void AllocateImpl(size_t size, void** ptr) {
    std::lock_guard<std::mutex> guard(lock_);
    auto iter = free_blocks_.lower_bound(size);
    if (iter != free_blocks_.end()) {
      *ptr = (iter->second);
      free_blocks_.erase(iter);
    } else {
      auto result = musaMalloc(ptr, size);
      TORCH_CHECK(
          result == ::musaError::musaSuccess, "Musa Tensor Allocate failed!");
      if (break_down_) {
        return;
      }
      blocks_.emplace(std::make_pair(*ptr, size));
      current_memory_in_use_ += size;
      break_down_ = (current_memory_in_use_ >= max_memory_in_bytes_);
    }
  }

  void FreeImpl(void* ptr) {
    std::lock_guard<std::mutex> guard(lock_);
    auto it = blocks_.find(ptr);
    TORCH_INTERNAL_ASSERT(
        it != blocks_.end() || break_down_, "ptr can not found.");
    if (it == blocks_.end()) {
      auto result = musaFree(ptr);
      TORCH_CHECK(
          result == ::musaError::musaSuccess, "Musa Tensor Release failed!");
    } else {
      free_blocks_.emplace(it->second, it->first);
    }
  }

 private:
  AutoGrowthBestFitAllocator() : break_down_(false), current_memory_in_use_(0) {
    const char* env = getenv("MTGPU_MAX_MEM_USAGE_GB");
    if (env == nullptr) {
      max_memory_in_bytes_ = 16;
    } else {
      max_memory_in_bytes_ = std::stoull(std::string(env));
    }
    max_memory_in_bytes_ = max_memory_in_bytes_ << 30;
  }

  ~AutoGrowthBestFitAllocator() {
    for (auto b : blocks_) {
      auto result = musaFree(b.first);
      TORCH_CHECK(
          result == ::musaError::musaSuccess, "Musa Tensor Release failed!");
    }
  }

  bool break_down_;
  uint64_t current_memory_in_use_;
  uint64_t max_memory_in_bytes_;
  std::map<void*, size_t> blocks_;
  std::multimap<size_t, void*> free_blocks_;
  std::mutex lock_;
};
} // namespace musa
