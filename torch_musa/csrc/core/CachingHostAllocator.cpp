#include <musa_runtime_api.h>
#include <stdint.h>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <ATen/DeviceGuard.h>
//#include <c10/core/DeviceGuard.h>

#include "torch_musa/csrc/aten/musa/Exceptions.h"
#include "torch_musa/csrc/core/CachingHostAllocator.h"
#include "torch_musa/csrc/core/MUSAEvent.h"
#include "torch_musa/csrc/core/MUSAFunctions.h"
#include "torch_musa/csrc/core/MUSAHooks.h"
#include "torch_musa/csrc/core/MUSAHooksInterface.h"

namespace at {
namespace musa {
namespace {

struct BlockSize {
  size_t size_{0};
  void* ptr_{nullptr};
};

struct Block {
  size_t size_{0};
  void* ptr_{nullptr};

  std::mutex mutex_;
  bool allocated_{false};
  size_t event_count_{0};
  std::unordered_set<at::musa::MUSAStream> streams_;
};

// Note: musaEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling musaEventCreate/musaEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<
      at::musa::MUSAEvent,
      std::function<void(at::musa::MUSAEvent*)>>;
  EventPool() : pools_(at::musa::device_count()) {}

  Event get(DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<DeviceIndex>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](at::musa::MUSAEvent* event) {
      std::lock_guard<std::mutex> guard(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<at::musa::MUSAEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> guard(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    return Event(
        std::make_unique<at::musa::MUSAEvent>(musaEventDisableTiming).release(),
        destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> guard(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<at::musa::MUSAEvent>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

// Used for heterogeneous lookup support in the free list.
struct BlockComparator {
  using is_transparent = void;
  bool operator()(const Block* a, const Block* b) const {
    if (a->size_ != b->size_) {
      return a->size_ < b->size_;
    }
    return (uintptr_t)a->ptr_ < (uintptr_t)b->ptr_;
  }

  // Transparent overloads
  bool operator()(const Block* a, BlockSize b) const {
    if (a->size_ != b.size_) {
      return a->size_ < b.size_;
    }
    return (uintptr_t)a->ptr_ < (uintptr_t)b.ptr_;
  }
  bool operator()(BlockSize a, const Block* b) const {
    if (a.size_ != b->size_) {
      return a.size_ < b->size_;
    }
    return (uintptr_t)a.ptr_ < (uintptr_t)b->ptr_;
  }
};

/**
 * Note [MUSAHostAllocator design]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * We have three key data structures - the free list which stores blocks that
 * are not currently used, the block list which stores all blocks that have been
 * allocated, and the event queue which stores MUSA events and their
 * corresponding blocks.
 *
 * Each of these are protected by a separate mutex. The key design principles
 * are to 1) only hold each mutex for the minimal amount of time possible, 2)
 * never do any possible expensive operations (such as MUSA runtime API calls)
 * while holding the lock.
 *
 * There are three public methods: allocate, free, and record_event. In the
 * allocate path, we first check to see if we can service our request from this
 * free list, and otherwise we create a new block with musaHostAlloc. In the
 * free path, we insert events (if required) into the event queue, and if
 * possible insert our block back into the free list. In allocate, we first
 * eagerly query events until we find one that is not ready, and insert the
 * corresponding block onto the free list if all the events recorded for a
 * block are ready. In the record_event path, we simply insert the given
 * stream into the set of streams tracked by the specified block. This set of
 * streams is then consumed in the free path.
 *
 * Some of the invariants here are less strict than they could be - for example,
 * we do not enforce that free(Block* block) => block->event_count == 0. This is
 * for compatibility reasons, and we can explore enforcing these in subsequent
 * versions.
 */
class MUSAHostAllocator {
 public:
  std::pair<void*, void*> allocate(size_t size) {
    if (size == 0) {
      return {nullptr, nullptr};
    }

    process_events();

    // First, try to allocate from the free list
    {
      std::lock_guard<std::mutex> g(free_list_mutex_);
      auto it = free_list_.lower_bound(BlockSize{size, nullptr});
      if (it != free_list_.end()) {
        auto block = *it;
        block->allocated_ = true;
        free_list_.erase(it);
        return {block->ptr_, reinterpret_cast<void*>(block)};
      }
    }
    // Then, create a new block.
    // Pinned memory pointers allocated by any device can be directly used by
    // any other device, regardless of the current device at the time of
    // allocation, since we assume unified addressing. So we grab any existing
    // primary context, if available. See pytorch/pytorch#21081.
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index =
        c10::musa::getDeviceIndexWithPrimaryContext();
    // device_guard can only be initialized by follow ctx, it must has_value,
    // or device_guard will call Destructor with invalid property.
    assert(primary_ctx_device_index.has_value());
    device_guard.reset_device(
        at::Device(at::musa::kMUSA, *primary_ctx_device_index));

    // Round up the allocation to the nearest power of two to improve reuse.
    // TODO(MTAI): We might want to employ more efficient approaches, such as
    // memory coalescing for better pinned memory management.
    void* ptr = nullptr;
    TORCH_MUSA_CHECK(musaHostAlloc(
        &ptr, c10::llvm::PowerOf2Ceil(size), musaHostAllocDefault));
    auto block = new Block();
    block->size_ = c10::llvm::PowerOf2Ceil(size);
    block->ptr_ = ptr;
    block->allocated_ = true;

    {
      std::lock_guard<std::mutex> g(blocks_mutex_);
      blocks_.insert(block);
      ptr_to_block_.insert({block->ptr_, block});
    }
    return {block->ptr_, reinterpret_cast<void*>(block)};
  }

  void free(void* ctx) {
    if (!ctx) {
      return;
    }

    // Note: we can assume that free is correctly paired with alloc,
    // and thus we do not need to look up the ctx in blocks_.
    auto* block = reinterpret_cast<Block*>(ctx);

    c10::optional<std::vector<EventPool::Event>> events;
    {
      std::lock_guard<std::mutex> guard(block->mutex_);
      block->allocated_ = false;
      if (block->streams_.empty()) {
        TORCH_INTERNAL_ASSERT(block->event_count_ == 0);
      } else {
        events = std::vector<EventPool::Event>();
        events->reserve(block->streams_.size());
        for (auto stream : block->streams_) {
          auto event = event_pool_.get(stream.device_index());
          event->record(stream);
          events->push_back(std::move(event));
        }
        block->event_count_ += events->size();
        block->streams_.clear();
      }
    }

    if (!events) {
      std::lock_guard<std::mutex> g(free_list_mutex_);
      free_list_.insert(block);
    } else {
      std::lock_guard<std::mutex> g(musa_events_mutex_);
      for (auto&& event : *events) {
        musa_events_.emplace_front(std::move(event), block);
      }
    }
  }

  bool record_event(void* ptr, void* ctx, at::musa::MUSAStream stream) {
    auto* block = reinterpret_cast<Block*>(ctx);

    // Note: we need to check if the passed-in `ctx` is valid. This is because
    // `record_event` (via `CachingHostAllocator_recordEvent`) can be invoked on
    // an arbitrary tensor, and is not guaranteed to correspond to a pinned
    // memory allocation. Therefore, we need to check that `ctx` is valid before
    // proceeding.
    {
      std::lock_guard<std::mutex> guard(blocks_mutex_);
      if (blocks_.find(block) != blocks_.end()) {
        // Now we know this object is safe to access.
        std::lock_guard<std::mutex> guard_block(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
      auto it = ptr_to_block_.find(ptr);
      if (it != ptr_to_block_.end()) {
        block = it->second;
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
    }

    return false;
  }

  void empty_cache() {
    // Flush any available blocks into the free_list.
    process_events();

    // Release cached events from the event pool.
    event_pool_.empty_cache();

    // Remove all elements from the free list, remove them from the blocks
    // list, and free the associated pinned memory allocation. This requires
    // concurrently holding both the free list mutex and the blocks mutex, and
    // is the only function that concurrently holds multiple mutexes.
    std::lock(free_list_mutex_, blocks_mutex_);
    std::lock_guard<std::mutex> guard_free_list(
        free_list_mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> guard_block(blocks_mutex_, std::adopt_lock);

    std::vector<Block*> blocks_to_remove(free_list_.begin(), free_list_.end());
    free_list_.clear();
    for (auto* block : blocks_to_remove) {
      blocks_.erase(block);
      ptr_to_block_.erase(block->ptr_);
      AT_MUSA_CHECK(musaFreeHost(block->ptr_));
      delete block;
    }
  }

 private:
  void process_events() {
    while (true) {
      // Avoid calling musaEventDestroy while holding a mutex, so move
      // intermediate events out of the lock into this object.
      c10::optional<std::pair<EventPool::Event, Block*>> processed;

      {
        std::lock_guard<std::mutex> guard(musa_events_mutex_);
        if (!musa_events_.empty()) {
          processed = std::move(musa_events_.back());
          musa_events_.pop_back();
        }
      }

      if (!processed) {
        return;
      }

      // otherwise, query the event
      {
        // now, see if we can handle this element
        auto& event = processed->first;
        musaError_t err = musaEventQuery(*event);
        if (err == musaErrorNotReady) {
          (void)musaGetLastError(); // clear MUSA error
          // push the event onto the back of the queue if it's not
          // ready. TODO: do we need some debouncing logic to avoid allocating
          // threads repeatedly spinning on an event?
          {
            std::lock_guard<std::mutex> guard(musa_events_mutex_);
            musa_events_.push_back(std::move(*processed));
          }
          return;
        } else if (err != musaSuccess) {
          TORCH_MUSA_CHECK(err);
        }
      }

      // Process the events.
      TORCH_INTERNAL_ASSERT(processed);
      auto* block = processed->second;
      bool available = false;
      {
        std::lock_guard<std::mutex> guard(block->mutex_);
        TORCH_INTERNAL_ASSERT(!block->allocated_)
        block->event_count_--;
        if (block->event_count_ == 0) {
          available = true;
        }
      }

      if (available) {
        std::lock_guard<std::mutex> guard(free_list_mutex_);
        free_list_.insert(block);
      }
    }
  }

  EventPool event_pool_;

  alignas(64) std::mutex blocks_mutex_;
  std::unordered_set<Block*> blocks_;
  std::unordered_map<void*, Block*> ptr_to_block_;
  // Note: sharding this mutex seems to be profitable in heavily multi-threaded
  // scenarios.
  alignas(64) std::mutex free_list_mutex_;
  // Note: an alternative datastructure can yield significant wins here in
  // microbenchmarks.
  std::set<Block*, BlockComparator> free_list_;

  alignas(64) std::mutex musa_events_mutex_;
  std::deque<std::pair<EventPool::Event, Block*>> musa_events_;
};

} // namespace

static MUSAHostAllocator& getMUSAHostAllocator() {
  // leak and don't worry about shutdown
  static auto* r = new MUSAHostAllocator();
  return *r;
}

static void MUSAHostAllocatorDeleter(void* ctx) {
  getMUSAHostAllocator().free(ctx);
}

bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    at::musa::MUSAStream stream) {
  return getMUSAHostAllocator().record_event(ptr, ctx, stream);
}

// Releases cached pinned memory allocations via musaHostFree
void CachingHostAllocator_emptyCache() {
  getMUSAHostAllocator().empty_cache();
}

struct MUSAHostAllocatorWrapper final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    auto ptr_and_ctx = getMUSAHostAllocator().allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &MUSAHostAllocatorDeleter,
        at::DeviceType::CPU};
  }
};

static MUSAHostAllocatorWrapper musa_host_allocator;

at::Allocator* getCachingHostAllocator() {
  return &musa_host_allocator;
}

} // namespace musa
} // namespace at
