#include "torch_musa/csrc/core/CachingHostAllocator.h"

#include <future>

#include <ATen/DeviceGuard.h>

#include "torch_musa/csrc/aten/musa/Exceptions.h"
#include "torch_musa/csrc/core/MUSAAllocatorConfig.h"
#include "torch_musa/csrc/core/Pool.h"

namespace at::musa {
namespace {

using EventPool = detail::EventPool<MUSAEvent, true>;

using Block = HostBlock<MUSAStream>;

struct MUSACachingHostAllocatorImpl
    : public CachingHostAllocatorImpl<MUSAStream, EventPool::Event> {
 private:
  void allocate_host_memory(size_t size, void** ptr) override {
    // Pinned memory pointers allocated by any device can be directly used by
    // any other device, regardless of the current device at the time of
    // allocation, since we assume unified addressing. So we grab any existing
    // primary context, if available. See pytorch/pytorch#21081.
    // This can be a large performance hit if we cross NUMA nodes by allocating
    // and pinning memory on one side of the NUMA node and then using it on the
    // other side. Thankfully, we use one process per GPU, so we don't run into
    // this issue.
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index =
        c10::musa::getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      device_guard.reset_device(
          at::Device(kMUSA, *primary_ctx_device_index));
    }

    auto start = std::chrono::system_clock::now();
    if (c10::musa::MUSACachingAllocator::MUSAAllocatorConfig::
            pinned_use_musa_host_register()) {
      allocWithMusaHostRegister(ptr, size);
    } else {
      // Use musaHostAlloc for allocating pinned memory (global lock in driver)
      C10_MUSA_CHECK(musaHostAlloc(ptr, size, musaHostAllocDefault));
    }
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update the statistics on the time spent on musaHostAlloc/hostRegister
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      stats_.host_alloc_time.increase(duration.count());
    }
  }

  void free_block(Block* block) override {
    auto start = std::chrono::system_clock::now();
    if (c10::musa::MUSACachingAllocator::MUSAAllocatorConfig::
            pinned_use_musa_host_register()) {
      void* ptr = block->ptr_;
      AT_MUSA_CHECK(musaHostUnregister(ptr));
      std::free(ptr);
    } else {
      AT_MUSA_CHECK(musaFreeHost(block->ptr_));
    }
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update the statistics on the time spent on musaFreeHost/hostUnregister
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      stats_.host_free_time.increase(duration.count());
    }
  }

  void record_stream(
      std::optional<std::vector<EventPool::Event>>& events,
      MUSAStream stream) override {
    auto event = create_event_internal(stream.device_index());
    event->record(stream);
    events->push_back(std::move(event));
  }

  bool query_event(EventPool::Event& event) override {
    musaError_t err = musaEventQuery(*event);
    if (err == musaErrorNotReady) {
      (void)musaGetLastError(); // clear MUSA error
      return false;
    } else if (err != musaSuccess) {
      C10_MUSA_CHECK(err);
    }
    return true;
  }

  bool pinned_use_background_threads() override {
    return c10::musa::MUSACachingAllocator::MUSAAllocatorConfig::
        pinned_use_background_threads();
  }

  EventPool::Event create_event_internal(DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issue.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  TaskThreadPool* getThreadPool() {
    static TaskThreadPool* pool = new TaskThreadPool(
        static_cast<int>(c10::musa::MUSACachingAllocator::MUSAAllocatorConfig::
            pinned_max_register_threads()));
    return pool;
  }

  void mapPagesForRegister(
      const void* ptr,
      size_t size,
      size_t i,
      size_t numThreads,
      size_t pageSize) {
    uintptr_t start = (uintptr_t)ptr + (size * i / numThreads);
    uintptr_t end = (uintptr_t)start + (size / numThreads);
    if (i == (numThreads - 1)) {
      end = (uintptr_t)ptr + size;
    }

    // pre-fault/map the pages by setting the first byte of the page
    uintptr_t alignedStart =
        (((uintptr_t)start + pageSize - 1) & ~(pageSize - 1));
    for (uintptr_t p = alignedStart; p < ((uintptr_t)end); p += pageSize) {
      memset((void*)p, 0, 1);
    }
  }

  void registerPages(const void* ptr, size_t size) {
    AT_MUSA_CHECK(
        musaHostRegister((void*)ptr, (size_t)size, musaHostRegisterDefault));

    // If host and device pointer don't match, give a warning and exit
    void* devptr = nullptr;
    AT_MUSA_CHECK(musaHostGetDevicePointer(&devptr, (void*)ptr, 0));
    TORCH_CHECK(
        (void*)devptr == (void*)ptr,
        "Host and device pointer dont match with musaHostRegister. "
        "Please dont use this feature by setting "
        "PYTORCH_MUSA_ALLOC_CONF=use_musa_host_register:False (default)",
        "");
  }

  void allocWithMusaHostRegister(void** ptr, size_t roundSize) {
    // Here we do regular allocation, pre-fault/map the pages, and then do
    // musaHostRegister with GPU mapping flags to lock the pages, so we
    // can minimize the cost for the musa global lock.
    *ptr = std::malloc(roundSize);

    // Parallelize the mapping/registering of pages to reduce wall time
    size_t pageSize = (1 << 12); // 4kB pages
    size_t numMapThreads = c10::musa::MUSACachingAllocator::
        MUSAAllocatorConfig::pinned_num_register_threads();
    if ((numMapThreads > 1) && (roundSize >= (pageSize * numMapThreads))) {
      // parallelize the mapping of pages with a threadpool
      auto* pool = getThreadPool();
      std::vector<std::promise<void>> promises;
      std::vector<std::future<void>> futures;
      promises.reserve(numMapThreads);
      futures.reserve(numMapThreads);

      for (size_t i = 0; i < numMapThreads; i++) {
        promises.emplace_back();
        futures.push_back(promises[i].get_future());
        auto task = [this,
                     i,
                     ptr,
                     roundSize,
                     numMapThreads,
                     pageSize,
                     &promises]() mutable {
          mapPagesForRegister(
              *ptr,
              roundSize,
              i, // thread task-id
              numMapThreads,
              pageSize);
          // set the promise when mapping pages are done
          promises[i].set_value();
        };
        pool->run(task);
      }
      for (auto& future : futures) {
        future.wait();
      }
    } else {
      // Map pages in the same thread
      mapPagesForRegister(*ptr, roundSize, 0, 1, pageSize);
    }

    // Register the mapped pages using musaHostRegister
    registerPages(*ptr, roundSize);
  }
};

void raw_local_deleter(void* ptr);

struct MUSACachingHostAllocator final
    : public CachingHostAllocatorInterface<MUSACachingHostAllocatorImpl> {
  at::DataPtr allocate(size_t size) override {
    auto ptr_and_ctx = impl_->allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &raw_local_deleter,
        at::DeviceType::CPU};
  }
};

MUSACachingHostAllocator caching_host_allocator;

MUSACachingHostAllocator& getMUSACachingHostAllocator() {
  return caching_host_allocator;
}

void raw_local_deleter(void* ptr) {
  getMUSACachingHostAllocator().free(ptr);
}

} // anonymous namespace

bool CachingHostAllocator_recordEvent(void* ptr, void* ctx, MUSAStream stream) {
  return getMUSACachingHostAllocator().record_event(ptr, ctx, stream);
}

// Releases cached pinned memory allocations via musaHostFree
void CachingHostAllocator_emptyCache() {
  getMUSACachingHostAllocator().empty_cache();
}

at::Allocator* getCachingHostAllocator() {
  return &getMUSACachingHostAllocator();
}

at::HostStats CachingHostAllocator_getStats() {
  return getMUSACachingHostAllocator().getStats();
}

void CachingHostAllocator_resetAccumulatedStats() {
  return getMUSACachingHostAllocator().resetAccumulatedStats();
}

void CachingHostAllocator_resetPeakStats() {
  return getMUSACachingHostAllocator().resetPeakStats();
}

} // namespace at::musa
