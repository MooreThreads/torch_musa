#include "torch_musa/csrc/core/Allocator.h"

#include <mudnn.h>

#include <regex>

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/utils/Logging.h"

namespace c10 {

C10_DEFINE_REGISTRY(FreeMusaMemoryCallbacksRegistry, FreeMemoryCallback);

namespace musa {
namespace MUSACachingAllocator {

//
// Yet another caching allocator for MUSA device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to musaMalloc.
// - If the musaMalloc fails, the allocator will attempt to free one cached
//   block of sufficient size that is not split and retry the allocation.
//   If this also fails, the allocator will attempt to free all cached blocks
//   that are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using musaMalloc.
// - To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
// - To further reduce fragmentation, blocks >= 200MB are not allowed to be
//   split. These oversize cached blocks will still satisfy requests within
//   20MB of the oversize cached block size.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
// work.
//

namespace {

// TODO(yang.zhao):
//    1. Add members and functions to support MUSA graphs, events and streams.
//    2. Modify block-related hyper-params to fit MTGPU (need experiments).
//    3. Add a allocator-vector to manage multiple devices.
//    4. Add c10::reportMemoryUsageToProfiler() when python API allows.

// all sizes are rounded to at least 512 bytes
constexpr size_t kMinBlockSize = 512;
// largest "small" allocation is 1 MiB
constexpr size_t kMaxSmallAlloc = 1048576;
// "small" allocations are packed in 2 MiB blocks
constexpr size_t kSmallBuffer = 2097152;
// "large" allocations may be packed in 20 MiB blocks
constexpr size_t kLargeBuffer = 20971520;
// allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kMinLargeAlloc = 10485760;
// round up large allocations to 2 MiB
constexpr size_t kRoundLarge = 2097152;

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

void update_stat(Stat& stat, int64_t amount) {
  stat.current += amount;

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stat.current >= 0,
      "Negative tracked stat in MUSA allocator (likely logic error).");

  stat.peak = std::max(stat.current, stat.peak);
  if (amount > 0) {
    stat.allocated += amount;
  }
  if (amount < 0) {
    stat.freed += -amount;
  }
}

void reset_accumulated_stat(Stat& stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

void reset_peak_stat(Stat& stat) {
  stat.peak = stat.current;
}

template <typename Func>
void for_each_selected_stat_type(const StatTypes& stat_types, Func f) {
  for (const auto stat_type : c10::irange(stat_types.size())) {
    if (stat_types[stat_type]) {
      f(stat_type);
    }
  }
}

void update_stat_array(
    StatArray& stat_array,
    int64_t amount,
    const StatTypes& stat_types) {
  for_each_selected_stat_type(
      stat_types, [&stat_array, amount](size_t stat_type) {
        update_stat(stat_array[stat_type], amount);
      });
}

struct Block;
typedef bool (*Comparison)(const Block*, const Block*);

struct BlockPool {
  BlockPool(Comparison comparator, bool small)
      : blocks(comparator), is_small(small) {}
  std::set<Block*, Comparison> blocks;
  const bool is_small;
};

struct Block {
  int device; // gpu
  size_t size; // block size in bytes
  BlockPool* pool; // owning memory pool
  void* ptr; // memory address
  bool allocated; // in-use flag
  Block* prev; // prev block if split from a larger allocation
  Block* next; // next block if split from a larger allocation
  int gc_count; // counter for prioritizing older / less useful blocks for
                // garbage collection

  Block(int device, size_t size, BlockPool* pool, void* ptr)
      : device(device),
        size(size),
        pool(pool),
        ptr(ptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        gc_count(0) {}

  // constructor for search key
  Block(int device, size_t size)
      : device(device),
        size(size),
        pool(nullptr),
        ptr(nullptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        gc_count(0) {}

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
};

static bool BlockComparator(const Block* a, const Block* b) {
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

static std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

struct AllocParams {
  AllocParams(int device, size_t size, BlockPool* pool, size_t alloc_size)
      : search_key(device, size),
        pool(pool),
        alloc_size(alloc_size),
        block(nullptr),
        err(musaSuccess) {}

  int device() const {
    return search_key.device;
  }
  size_t size() const {
    return search_key.size;
  }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block;
  StatTypes stat_types = {false};
  musaError_t err;
};

} // namespace

class CachingAllocatorConfig {
 public:
  static size_t max_split_size() {
    return instance().m_max_split_size_;
  }
  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold_;
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As ane example, if we want 4 divisions between 2's power, this can be done
  // using env variable: PYTORCH_MUSA_ALLOC_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions() {
    return instance().m_roundup_power2_divisions_;
  }

 private:
  static CachingAllocatorConfig& instance() {
    static CachingAllocatorConfig* s_instance = ([]() {
      auto inst = new CachingAllocatorConfig();
      inst->parse_args();
      return inst;
    })();
    return *s_instance;
  }

  CachingAllocatorConfig()
      : m_max_split_size_(std::numeric_limits<size_t>::max()),
        m_roundup_power2_divisions_(0),
        m_garbage_collection_threshold_(0) {}
  // Set largest block size for splitting, remaining some large blocks not being
  // split
  size_t m_max_split_size_;
  // Set to round-up block size to nearest power of 2 divisions. See
  // roundup_power2_divisions()
  size_t m_roundup_power2_divisions_;
  // Set to affect how much portion of memory being considered to collect when
  // malloc fails.
  double m_garbage_collection_threshold_;

  // yang.zhao: An example for the env var PYTORCH_MUSA_ALLOC_CONF.
  // PYTORCH_MUSA_ALLOC_CONF=max_split_size_mb:20,garbage_collection_threshold:0.5
  void parse_args() {
    const char* val = getenv("PYTORCH_MUSA_ALLOC_CONF");
    if (val != NULL) {
      const std::string config(val);

      std::regex exp("[\\s,]+");
      std::sregex_token_iterator it(config.begin(), config.end(), exp, -1);
      std::sregex_token_iterator end;
      std::vector<std::string> options(it, end);

      for (auto option : options) {
        std::regex exp2("[:]+");
        std::sregex_token_iterator it2(option.begin(), option.end(), exp2, -1);
        std::sregex_token_iterator end2;
        std::vector<std::string> kv(it2, end2);
        if (kv.size() >= 2) {
          /* Maximum split size in MB.  Limited to large size blocks */
          if (kv[0].compare("max_split_size_mb") == 0) {
            size_t val2 = stoi(kv[1]);
            TORCH_CHECK(
                val2 > kLargeBuffer / (1024 * 1024),
                "CachingAllocator option max_split_size_mb too small, must be > ",
                kLargeBuffer / (1024 * 1024),
                "");
            val2 = std::max(val2, kLargeBuffer / (1024 * 1024));
            val2 = std::min(
                val2, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
            m_max_split_size_ = val2 * 1024 * 1024;
          } else if (kv[0].compare("roundup_power2_divisions") == 0) {
            size_t val2 = stoi(kv[1]);
            TORCH_CHECK(
                c10::llvm::isPowerOf2_64(val2),
                "For roundups, the divisons has to be power of 2 ",
                "");
            m_roundup_power2_divisions_ = val2;
          } else if (kv[0].compare("garbage_collection_threshold") == 0) {
            /*
             * Perform garbage collection of GPU memory blocks to avoid
             * triggering expensive sync-and-reclaim-all operation. Upon setting
             * the threshold (e.g., 0.8), the allocator will start reclaiming
             * blocks if GPU memory capacity usage exceeds the threshold (i.e.,
             * 80% of total memory).
             * Values 0.0 and 1.0 are not allowed as they are less meaningful.
             */
            double val2 = stod(kv[1]);
            TORCH_CHECK(
                val2 > 0,
                "garbage_collect_threshold too small, set it 0.0~1.0",
                "");
            TORCH_CHECK(
                val2 < 1.0,
                "garbage_collect_threshold too big, set it 0.0~1.0",
                "");
            m_garbage_collection_threshold_ = val2;
          } else {
            TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", kv[0]);
          }
        }
      }
    }
  }
};

class MTGPUCachingAllocator {
 private:
  // lock around all operations
  mutable std::recursive_mutex mutex_;

  // device statistics
  DeviceStats stats_;

  // unallocated cached blocks larger than 1 MB
  BlockPool large_blocks_;

  // unallocated cached blocks 1 MB or smaller
  BlockPool small_blocks_;

  // allocated or in use by a stream. Holds all active allocations,
  // whether they came from graph_pools or one of the BlockPools above.
  ska::flat_hash_set<Block*> active_blocks_;

  // record used memory.
  size_t total_allocated_memory_ = 0;

  // if set_fraction is true, allowed_memory_maximum is the maximal
  // memory allowed to use
  size_t allowed_memory_maximum_ = 0;
  bool set_fraction_ = false;

  // device ID of this allocator
  int self_device_;

 public:
  MTGPUCachingAllocator()
      : large_blocks_(BlockComparator, /*is_small=*/false),
        small_blocks_(BlockComparator, /*is_small=*/true) {
    stats_.max_split_size = CachingAllocatorConfig::max_split_size();
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.

  Block* malloc(int device, size_t size) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);

    self_device_ = device;
    size = round_size(size);
    BlockPool& pool = get_pool(size);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, &pool, alloc_size);
    params.stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    params.stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;

    // First, try to get a block from the existing pool.
    bool block_found =
        // Search pool
        get_free_block(params)
        // Trigger callbacks and retry search
        || (trigger_free_memory_callbacks() && get_free_block(params));

    // Can't reuse an existing block; try to get a new one.
    if (!block_found) {
      // Do garbage collection if the flag is set.
      if (C10_UNLIKELY(
              set_fraction_ &&
              CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        garbage_collect_cached_blocks();
      }
      // Attempt allocate
      block_found = alloc_block(params, false)
          // Free enough available cached blocks to satisfy alloc and retry
          // alloc.
          || (release_available_cached_blocks(params) &&
              alloc_block(params, false))
          // Free all non-split cached blocks and retry alloc.
          || (release_cached_blocks() && alloc_block(params, true));
    }

    if (!block_found) {
      // For any error code other than musaErrorMemoryAllocation,
      // alloc_block should have thrown an exception already.
      TORCH_INTERNAL_ASSERT(params.err == musaErrorMemoryAllocation);

      size_t device_free;
      size_t device_total;
      TORCH_CHECK(
          musaMemGetInfo(&device_free, &device_total) == musaSuccess,
          "MUSA get memory info fails!");
      std::string allowed_info;

      if (set_fraction_) {
        allowed_info = format_size(allowed_memory_maximum_) + " allowed; ";
      }

      stats_.num_ooms += 1;

      // "total capacity": total global memory on GPU
      // "allowed": memory is allowed to use, which set by fraction.
      // "already allocated": memory allocated by the program using the
      //                      caching allocator
      // "free": free memory as reported by the MUSA API
      // "cached": memory held by the allocator but not used by the program
      //
      // The "allocated" amount  does not include memory allocated outside
      // of the caching allocator, such as memory allocated by other programs
      // or memory held by the driver.
      //
      // The sum of "allocated" + "free" + "cached" may be less than the
      // total capacity due to memory held by the driver and usage by other
      // programs.
      //
      // Note that at this point free_cached_blocks has already returned all
      // possible "cached" memory to the driver. The only remaining "cached"
      // memory is split from a larger block that is partially in-use.
      TORCH_CHECK_WITH(
          MUSAOutOfMemoryError,
          false,
          "MUSA out of memory. Tried to allocate ",
          format_size(alloc_size),
          " (GPU ",
          device,
          "; ",
          format_size(device_total),
          " total capacity; ",
          format_size(
              stats_.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                  .current),
          " already allocated; ",
          format_size(device_free),
          " free; ",
          allowed_info,
          format_size(
              stats_.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
                  .current),
          " reserved in total by PyTorch)",
          " If reserved memory is >> allocated memory try setting max_split_size_mb to avoid"
          " fragmentation.  See documentation for Memory Management and PYTORCH_MUSA_ALLOC_CONF",
          "");
    }

    TORCH_INTERNAL_ASSERT(
        params.err == musaSuccess && params.block != nullptr &&
        params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

    const bool already_split = block->is_split();
    if (should_split(block, size)) {
      remaining = block;

      block = new Block(device, size, &pool, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      bool inserted = pool.blocks.insert(remaining).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

      if (already_split) {
        // An already-split inactive block is being shrunk by size bytes.
        update_stat_array(
            stats_.inactive_split_bytes, -block->size, params.stat_types);
      } else {
        // A new split inactive block is being created from a previously unsplit
        // block, size remaining->size bytes.
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
          update_stat(stats_.inactive_split_bytes[stat_type], remaining->size);
          update_stat(stats_.inactive_split[stat_type], 1);
        });
      }
    } else if (already_split) {
      // An already-split block is becoming active
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        update_stat(stats_.inactive_split_bytes[stat_type], -block->size);
        update_stat(stats_.inactive_split[stat_type], -1);
      });
    }

    block->allocated = true;
    bool inserted = active_blocks_.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      update_stat(stats_.allocation[stat_type], 1);
      update_stat(stats_.allocated_bytes[stat_type], block->size);
      update_stat(stats_.active[stat_type], 1);
      update_stat(stats_.active_bytes[stat_type], block->size);
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats_.oversize_allocations, 1);

    return block;
  }

  void free(Block* block) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    block->allocated = false;

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] =
        true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats_.allocation[stat_type], -1);
      update_stat(stats_.allocated_bytes[stat_type], -block->size);
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats_.oversize_allocations, -1);

    free_block(block);
  }

  // This function is used to return the base ptr and the offset
  // when being called by python func.
  // (see torch/csrc/generic/StorageSharing.cpp:L268)
  void* get_base_allocation(Block* block, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    while (block->prev) {
      block = block->prev;
    }
    void* basePtr = block->ptr;
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  /** set memory fraction to limit maximum allocated memory **/
  void set_memory_fraction(double fraction) {
    size_t device_free;
    size_t device_total;
    TORCH_CHECK(
        musaMemGetInfo(&device_free, &device_total) == musaSuccess,
        "MUSA get memory info fails!");
    allowed_memory_maximum_ = static_cast<size_t>(fraction * device_total);
    set_fraction_ = true;
  }

  /** returns cached blocks to the system allocator **/
  void empty_cache() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    release_cached_blocks();
  }

  /** Retrieves info (total size + largest block) of the memory cache **/
  void cache_info(size_t* total, size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (*largest ==
        0) { // make an initial guess if a zero *largest is passed in
      size_t tmp_bytes;
      TORCH_CHECK(
          musaMemGetInfo(
              largest, // Use free memory as an optimistic initial guess of
                       // *largest
              &tmp_bytes) == musaSuccess,
          "MUSA get memory info fails!");
    }
    cache_info_aux(large_blocks_, total, largest);
    cache_info_aux(small_blocks_, total, largest);
  }

  /** Returns a copy of the memory allocator stats_ **/
  DeviceStats get_stats() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return stats_;
  }

  /** Resets the historical accumulation stats_ for the device **/
  void reset_accumulated_stats() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_accumulated_stat(stats_.allocation[statType]);
      reset_accumulated_stat(stats_.segment[statType]);
      reset_accumulated_stat(stats_.active[statType]);
      reset_accumulated_stat(stats_.inactive_split[statType]);
      reset_accumulated_stat(stats_.allocated_bytes[statType]);
      reset_accumulated_stat(stats_.reserved_bytes[statType]);
      reset_accumulated_stat(stats_.active_bytes[statType]);
      reset_accumulated_stat(stats_.inactive_split_bytes[statType]);
    }

    stats_.num_alloc_retries = 0;
    stats_.num_ooms = 0;
    reset_accumulated_stat(stats_.oversize_allocations);
    reset_accumulated_stat(stats_.oversize_segments);
  }

  /** Resets the historical peak stats_ for the device **/
  void reset_peak_stats() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_peak_stat(stats_.allocation[statType]);
      reset_peak_stat(stats_.segment[statType]);
      reset_peak_stat(stats_.active[statType]);
      reset_peak_stat(stats_.inactive_split[statType]);
      reset_peak_stat(stats_.allocated_bytes[statType]);
      reset_peak_stat(stats_.reserved_bytes[statType]);
      reset_peak_stat(stats_.active_bytes[statType]);
      reset_peak_stat(stats_.inactive_split_bytes[statType]);
    }
    reset_peak_stat(stats_.oversize_allocations);
    reset_peak_stat(stats_.oversize_segments);
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially
   * VERY expensive. **/
  std::vector<SegmentInfo> snapshot() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    std::vector<SegmentInfo> result;
    const std::vector<const Block*> all_blocks = get_all_blocks();

    for (const Block* const head_block : all_blocks) {
      if (head_block->prev != nullptr) {
        continue;
      }
      result.emplace_back();
      SegmentInfo& segment_info = result.back();
      segment_info.device = head_block->device;
      segment_info.address = reinterpret_cast<int64_t>(head_block->ptr);
      segment_info.is_large = (!head_block->pool->is_small);

      const Block* block = head_block;
      while (block != nullptr) {
        segment_info.blocks.emplace_back();
        BlockInfo& block_info = segment_info.blocks.back();

        block_info.size = block->size;
        block_info.allocated = block->allocated;
        block_info.active = block->allocated;

        segment_info.total_size += block_info.size;
        if (block_info.allocated) {
          segment_info.allocated_size += block_info.size;
        }
        if (block_info.active) {
          segment_info.active_size += block_info.size;
        }

        block = block->next;
      }
    }

    std::sort(
        result.begin(),
        result.end(),
        [](const SegmentInfo& a, const SegmentInfo& b) {
          return a.address < b.address;
        });

    return result;
  }

  // This function takes the size and number of divisions argument and rounds
  // up the size argument for the nearest power-of-2 division.
  // For example, if we need to round-up 1200 and number of divisions is 4,
  // the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
  // them, the values are 1024, 1280, 1536, and 1792. So the function will
  // return 1280 as the nearest ceiling of power-2 divison.
  static size_t roundup_power2_next_division(size_t size, size_t divisions) {
    if (C10_UNLIKELY(size <= 4 || divisions <= 1)) {
      return size;
    }
    if (c10::llvm::isPowerOf2_64(size)) {
      return size;
    }

    // divide the space between these 2's power into equal divisions
    // If division is zero, return the power-of-2 ceiling.
    size_t power2_floor = c10::llvm::PowerOf2Floor(size);
    size_t power2_divison =
        power2_floor >> (63 - c10::llvm::countLeadingZeros(divisions));
    if (C10_UNLIKELY(power2_divison == 0)) {
      return (power2_floor << 1);
    }
    size_t round_size_floor = size & (~(power2_divison - 1));
    return (round_size_floor == size) ? size
                                      : round_size_floor + power2_divison;
  }

  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      size_t divisions = CachingAllocatorConfig::roundup_power2_divisions();
      if (divisions > 0 && size > (kMinBlockSize * divisions)) {
        return roundup_power2_next_division(size, divisions);
      } else {
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
      }
    }
  }

 private:
  // All private methods do not acquire the allocator mutex.

  std::vector<const Block*> get_all_blocks() const {
    std::vector<const Block*> blocks;
    blocks.insert(
        blocks.end(), small_blocks_.blocks.begin(), small_blocks_.blocks.end());
    blocks.insert(
        blocks.end(), large_blocks_.blocks.begin(), large_blocks_.blocks.end());
    blocks.insert(blocks.end(), active_blocks_.begin(), active_blocks_.end());
    return blocks;
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(Block* block) {
    TORCH_INTERNAL_ASSERT(!block->allocated);

    size_t original_block_size = block->size;

    BlockPool& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      const int64_t subsumed_size =
          try_merge_blocks(block, merge_candidate, pool);
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    active_blocks_.erase(block);
    // Makes sure the Block* isn't already present in the pool we're freeing it
    // back into.
    bool inserted = pool.blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT(inserted);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += block->size;
    }

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(
          stats_.inactive_split[stat_type], net_change_inactive_split_blocks);
      update_stat(
          stats_.inactive_split_bytes[stat_type],
          net_change_inactive_split_size);
      update_stat(stats_.active[stat_type], -1);
      update_stat(stats_.active_bytes[stat_type], -original_block_size);
    });
  }

  /** combine previously split blocks. returns the size of the subsumed block,
   * or 0 on failure. */
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) {
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }

    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased = pool.blocks.erase(src);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete src;

    return subsumed_size;
  }

  BlockPool& get_pool(size_t size) {
    if (size <= kMaxSmallAlloc) {
      return small_blocks_;
    } else {
      return large_blocks_;
    }
  }

  StatType get_stat_type_for_pool(const BlockPool& pool) {
    return pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL;
  }

  bool should_split(const Block* block, size_t size) {
    // yang.zhao: if use default value (0), close the split feature
    if (0 == CachingAllocatorConfig::max_split_size()) {
      return false;
    }
    size_t remaining = block->size - size;
    if (block->pool->is_small) {
      return remaining >= kMinBlockSize;
    } else {
      return (size < CachingAllocatorConfig::max_split_size()) &&
          (remaining > kMaxSmallAlloc);
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kMaxSmallAlloc) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  bool get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;

    if (C10_UNLIKELY(
            set_fraction_ &&
            CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
      // Track block reuse interval only when garbage collection is enabled.
      for (auto& b : pool.blocks) {
        ++b->gc_count;
      }
    }
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end())
      return false;
    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer))
      return false;
    p.block = *it;
    (*it)->gc_count = 0; // Denote this block has been used
    pool.blocks.erase(it);
    return true;
  }

  bool trigger_free_memory_callbacks() {
    bool freed_memory = false;
    for (const auto& name : c10::FreeMusaMemoryCallbacksRegistry()->Keys()) {
      freed_memory |=
          c10::FreeMusaMemoryCallbacksRegistry()->Create(name)->Execute();
    }
    return freed_memory;
  }

  void garbage_collect_cached_blocks() {
    // Free unused cached blocks to reclaim GPU memory.
    // Unlike release_cached_blocks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    size_t gc_threshold = static_cast<size_t>(
        CachingAllocatorConfig::garbage_collection_threshold() *
        allowed_memory_maximum_);
    // No need to trigger GC yet
    if (total_allocated_memory_ <= gc_threshold) {
      return;
    }
    const size_t target_size = total_allocated_memory_ - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able blocks. We'll use it later to
    // get "avg age" threshold.
    double total_age = 0.0;
    int freeable_block_count = 0;
    for (auto& b : large_blocks_.blocks) {
      if (!b->is_split()) {
        total_age += b->gc_count;
        ++freeable_block_count;
      }
    }
    // No free-able blocks?
    if (freeable_block_count == 0) {
      return;
    }

    // Repeat GC until we reach reclaim > target size.
    bool block_freed = true;
    while (gc_reclaimed < target_size && block_freed == true &&
           freeable_block_count > 0) {
      // Free blocks exceeding this age threshold first.
      double age_threshold = total_age / freeable_block_count;
      // Stop iteration if we can no longer free a block.
      block_freed = false;

      // Free blocks of > avg age. Don't stop upon reaching the target_size,
      // we don't want this GC to be triggered frequently.
      auto it = large_blocks_.blocks.begin();
      while (it != large_blocks_.blocks.end()) {
        Block* block = *it;
        ++it;
        if (!block->is_split() && block->gc_count >= age_threshold) {
          block_freed = true;
          gc_reclaimed += block->size;
          total_age -= block->gc_count; // Decrement the age
          freeable_block_count--; // One less block that can be freed
          release_block(block);
        }
      }
    }
  }

  bool alloc_block(AllocParams& p, bool isRetry) {
    // Defensively checks for preexisting MUSA error state.
    TORCH_CHECK(
        musaGetLastError() == musaSuccess,
        "musaGetLastError() fails in alloc_block!");

    size_t size = p.alloc_size;
    void* ptr;

    if (isRetry) {
      stats_.num_alloc_retries += 1;
    }

    if (set_fraction_ &&
        total_allocated_memory_ + size > allowed_memory_maximum_) {
      p.err = musaErrorMemoryAllocation;
      return false;
    } else {
      p.err = musaMalloc(&ptr, size);
      if (p.err != musaSuccess) {
        if (p.err == musaErrorMemoryAllocation) {
          // If this is the first attempt (!isRetry), we can forgive and clear
          // MUSA's
          //   internal error state.
          // If this is the second attempt (isRetry), malloc's TORCH_CHECK_WITH
          // will take
          //   over to throw a helpful exception. The user can choose to catch
          //   the exception, free some stuff in their script, and attempt their
          //   allocation again. In this case, we can also forgive and clear
          //   MUSA's internal error state.
          musaGetLastError();
        } else {
          // If the error's unrelated to memory allocation, we should throw
          // immediately.
          TORCH_CHECK(p.err == musaSuccess, "Musa Tensor Allocate failed!");
        }
        return false;
      }
    }

    total_allocated_memory_ += size;
    p.block = new Block(p.device(), size, p.pool, (char*)ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      update_stat(stats_.segment[stat_type], 1);
      update_stat(stats_.reserved_bytes[stat_type], size);
    });
    if (size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats_.oversize_segments, 1);

    // p.block came from new, not musaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    return true;
  }

  /** Free one or more oversize blocks to the system allocator.  But only **/
  /** enough to satisfy the target size **/
  bool release_available_cached_blocks(const AllocParams& p) {
    if (CachingAllocatorConfig::max_split_size() ==
        std::numeric_limits<size_t>::max())
      return false;
    BlockPool& pool = *p.pool;
    Block key = p.search_key;
    key.size = (key.size < CachingAllocatorConfig::max_split_size())
        ? CachingAllocatorConfig::max_split_size()
        : key.size;
    auto it = pool.blocks.lower_bound(&key);
    if (it == pool.blocks.end()) {
      // No single block is large enough; free multiple oversize blocks,
      // starting with the largest
      if (it == pool.blocks.begin())
        return false;
      size_t totalReleased = 0;
      --it; // Back up one item.  Now on the largest block for the correct
            // stream
      while ((totalReleased < key.size) &&
             ((*it)->size >= CachingAllocatorConfig::max_split_size())) {
        auto cur = it;
        totalReleased += (*it)->size;
        if (it != pool.blocks.begin()) {
          --it;
          release_block(*cur);
        } else {
          release_block(*cur);
          break;
        }
      }
      if (totalReleased < key.size)
        return false;
    } else {
      release_block(*it);
    }
    return true;
  }

  bool release_cached_blocks() {
    // Free all non-split cached blocks to system allocator
    release_blocks(large_blocks_);
    release_blocks(small_blocks_);

    return true;
  }

  void release_block(Block* block) {
    auto err = musaFree((void*)block->ptr);
    TORCH_CHECK(err == musaSuccess, "Musa Tensor Release failed!");
    total_allocated_memory_ -= block->size;

    BlockPool* pool = block->pool;

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*pool))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats_.segment[stat_type], -1);
      update_stat(stats_.reserved_bytes[stat_type], -block->size);
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats_.oversize_segments, -1);

    pool->blocks.erase(block);
    delete block;
  }

  void release_blocks(BlockPool& pool) {
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (!block->prev && !block->next) {
        release_block(block);
      }
    }
  }

  // Accumulates sizes of all memory blocks for given device in given pool
  void cache_info_aux(const BlockPool& pool, size_t* total, size_t* largest) {
    for (const auto& block : pool.blocks) {
      const size_t blocksize = block->size;
      *total += blocksize;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }
};

} // namespace MUSACachingAllocator
} // namespace musa
} // namespace c10

namespace c10 {
namespace musa {

using namespace ::c10::musa::MUSACachingAllocator;

class MusaCachingAllocatorImpl {
 public:
  MusaCachingAllocatorImpl() {
    const int64_t dev_num = static_cast<int64_t>(c10::musa::device_count());
    device_allocators_.reserve(dev_num);
    for (int i = 0; i < dev_num; ++i) {
      device_allocators_.emplace_back(
          std::make_unique<MTGPUCachingAllocator>());
    }
  }

  Block* get_allocated_block(void* ptr, bool remove = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks.erase(it);
    }
    return block;
  }

  void malloc(void** devPtr, int device, size_t size) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocators_.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    Block* block = device_allocators_[device]->malloc(device, size);
    add_allocated_block(block);
    *devPtr = (void*)block->ptr;
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, true /* remove */);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    device_allocators_[block->device]->free(block);
  }

  void empty_cache() {
    for (auto& da : device_allocators_) {
      da->empty_cache();
    }
  }

  void reset_peak_stats() {
    for (auto& da : device_allocators_) {
      da->reset_peak_stats();
    }
  }

  std::vector<SegmentInfo> snapshot() {
    std::vector<SegmentInfo> result;
    for (auto& da : device_allocators_) {
      auto snap = da->snapshot();
      result.insert(result.end(), snap.begin(), snap.end());
    }

    return result;
  }

  DeviceStats get_stats(int64_t device) {
    return device_allocators_[device]->get_stats();
  }

  std::vector<std::unique_ptr<MTGPUCachingAllocator>> device_allocators_;

 private:
  std::mutex mutex_;

  // allocated blocks by device pointer
  ska::flat_hash_map<void*, Block*> allocated_blocks;

  void add_allocated_block(Block* block) {
    std::lock_guard<std::mutex> lock(mutex_);
    allocated_blocks[block->ptr] = block;
  }
};

struct C10_API MusaCachingAllocator final : at::Allocator {
 public:
  MusaCachingAllocator() {
    const char* env = getenv("MTGPU_ALLOC_TYPE");
    use_caching_allocator_ = true;
    if (env != nullptr && std::string(env) == "naive") {
      DLOG_INFO << "Use naive allocator.";
      use_caching_allocator_ = false;
    } else {
      DLOG_INFO << "Use caching allocator.";
      allocator_impl_ = new MusaCachingAllocatorImpl();
    }
  }

  at::DataPtr allocate(size_t nbytes) const override {
    void* data = nullptr;
    int device;
    TORCH_MUSA_CHECK(musaGetDevice(&device));
    if (nbytes) {
      if (use_caching_allocator_) {
        allocator_impl_->malloc(&data, device, nbytes);
      } else {
        ::c10::musa::AutoGrowthBestFitAllocator::get_allocator()->AllocateImpl(
            nbytes, &data);
      }
    }
    return {
        data, data, &raw_delete, at::Device(at::native::musa::kMUSA, device)};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &raw_delete;
  }

  MusaCachingAllocatorImpl* get_allocator() const {
    return allocator_impl_;
  }

  bool use_caching_allocator_;

 private:
  MusaCachingAllocatorImpl* allocator_impl_;
};

MusaCachingAllocator g_musa_alloc;

MusaCachingAllocator* GetMusaCachingAllocator() {
  return &g_musa_alloc;
}

void raw_delete(void* ptr) {
  MusaCachingAllocator* palloc = c10::musa::GetMusaCachingAllocator();
  palloc->get_allocator()->free(ptr);
}

REGISTER_ALLOCATOR(at::native::musa::kMUSA, &g_musa_alloc);

} // namespace musa
} // namespace c10

namespace c10 {
namespace musa {
namespace MUSACachingAllocator {

Allocator* get() {
  return c10::musa::GetMusaCachingAllocator();
}

void EmptyCache() {
  c10::musa::MusaCachingAllocator* palloc =
      c10::musa::GetMusaCachingAllocator();
  palloc->get_allocator()->empty_cache();
}

void ResetPeakStats() {
  c10::musa::MusaCachingAllocator* palloc =
      c10::musa::GetMusaCachingAllocator();
  palloc->get_allocator()->reset_peak_stats();
}

DeviceStats GetDeviceStats(int64_t device) {
  c10::musa::MusaCachingAllocator* palloc =
      c10::musa::GetMusaCachingAllocator();
  return palloc->get_allocator()->get_stats(device);
}

std::vector<SegmentInfo> GetMemorySnapshot() {
  c10::musa::MusaCachingAllocator* palloc =
      c10::musa::GetMusaCachingAllocator();
  return palloc->get_allocator()->snapshot();
}

} // namespace MUSACachingAllocator
} // namespace musa
} // namespace c10
