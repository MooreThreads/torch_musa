#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_MTGPUALLOCATOR_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_MTGPUALLOCATOR_H_

#include <ATen/ATen.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/DeviceType.h>
#pragma GCC diagnostic pop

#include <mudnn.h>
#include <musa_runtime.h>
#include <list>

// yang.zhao: Macros defined in c10/cuda/CUDAMacros.h,
//            replacing all CUDA with MUSA

// See c10/macros/Export.h for a detailed explanation of what the function
// of these macros are.  We need one set of macros for every separate library
// we build.

#ifdef _WIN32
#if defined(C10_MUSA_BUILD_SHARED_LIBS)
#define C10_MUSA_EXPORT __declspec(dllexport)
#define C10_MUSA_IMPORT __declspec(dllimport)
#else
#define C10_MUSA_EXPORT
#define C10_MUSA_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define C10_MUSA_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_MUSA_EXPORT
#endif // defined(__GNUC__)
#define C10_MUSA_IMPORT C10_MUSA_EXPORT
#endif // _WIN32

// This one is being used by libc10_cuda.so
#ifdef C10_MUSA_BUILD_MAIN_LIB
#define C10_MUSA_API C10_MUSA_EXPORT
#else
#define C10_MUSA_API C10_MUSA_IMPORT
#endif

// yang.zhao: predefined classes copied from CUDACachingAllocator.h

namespace c10 {

class C10_MUSA_API MUSAOutOfMemoryError : public c10::Error {
  using Error::Error;
};

// Caching allocator will execute every registered callback if it's unable to
// find a block inside of already allocated area.
class C10_MUSA_API FreeMemoryCallback {
 public:
  virtual ~FreeMemoryCallback() = default;
  virtual bool Execute() = 0;
};

C10_DECLARE_REGISTRY(FreeMusaMemoryCallbacksRegistry, FreeMemoryCallback);
#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(FreeMusaMemoryCallbacksRegistry, name, __VA_ARGS__);

namespace musa {

void raw_delete(void* ptr);

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
          result == ::musaError::musaSuccess, " Tensor Allocate failed!");
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
          result == ::musaError::musaSuccess, " Tensor Release failed!");
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
          result == ::musaError::musaSuccess, " Tensor Release failed!");
    }
  }

  bool break_down_;
  uint64_t current_memory_in_use_;
  uint64_t max_memory_in_bytes_;
  std::map<void*, size_t> blocks_;
  std::multimap<size_t, void*> free_blocks_;
  std::mutex lock_;
};

namespace MUSACachingAllocator {

struct Stat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

enum struct StatType : uint64_t {
  AGGREGATE = 0,
  SMALL_POOL = 1,
  LARGE_POOL = 2,
  NUM_TYPES = 3 // remember to update this whenever a new stat type is added
};

typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from musaMalloc().
  StatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be
  // released via musaFree)
  StatArray inactive_split;

  // SUM: bytes requested by client code
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;

  // COUNT: total number of failed calls to MUSA malloc necessitating cache
  // flushes.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to MUSA after cache flush)
  int64_t num_ooms = 0;

  // COUNT: total number of oversize blocks allocated from pool
  Stat oversize_allocations;

  // COUNT: total number of oversize blocks requiring malloc
  Stat oversize_segments;

  // SIZE: maximum block size that is allowed to be split.
  int64_t max_split_size = 0;
};

// Struct containing info of an allocation block (i.e. a fractional part of a
// musaMalloc)..
struct BlockInfo {
  int64_t size = 0;
  int32_t gc_counter = 0;
  bool allocated = false;
  bool active = false;
};

// Struct containing info of a memory segment (i.e. one contiguous musaMalloc).
struct SegmentInfo {
  int64_t device = 0;
  int64_t address = 0;
  int64_t total_size = 0;
  int64_t allocated_size = 0;
  int64_t active_size = 0;
  bool is_large = false;
  std::vector<BlockInfo> blocks;
};

void EmptyCache();
void ResetPeakStats();
DeviceStats GetDeviceStats(int64_t device);
std::vector<SegmentInfo> GetMemorySnapshot();

} // namespace MUSACachingAllocator
} // namespace musa
} // namespace c10
#endif // ATEN_SRC_ATEN_NATIVE_MUSA_MTGPUALLOCATOR_H_
