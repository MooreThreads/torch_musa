#pragma once

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/musa/MUSAMacros.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

#include "torch_musa/csrc/core/MUSAGraphsC10Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace c10 {

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
class C10_MUSA_API FreeMemoryCallback {
 public:
  virtual ~FreeMemoryCallback() = default;
  virtual bool Execute() = 0;
};

C10_DECLARE_REGISTRY(FreeMusaMemoryCallbacksRegistry, FreeMemoryCallback);
#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(FreeMusaMemoryCallbacksRegistry, name, __VA_ARGS__)
} // namespace c10
  //
// TODO: Turn this into an honest to goodness class. I briefly attempted to do
// this, but it was a bit irritating to figure out how to also correctly
// apply pimpl pattern so I didn't have to leak any internal implementation
// details in the header (MUSACachingAllocator could be made a pimpl, but
// you also need to appropriately define a class which is a subclass
// of Allocator. Not impossible, but required a bit more surgery than
// I wanted to do at the time.)
//
// Why is this using a namespace rather than old-style THMCachingAllocator_
// prefix?  Mostly because it made the HIPify rules easier to write; _ is
// not counted as a word boundary, so you would otherwise have to list each
// of these functions.

namespace c10::musa::MUSACachingAllocator {

// Preserved only for BC reasons
// NOLINTNEXTLINE(misc-unused-using-decls)
using c10::CachingDeviceAllocator::AllocatorTraceTracker;
using c10::CachingDeviceAllocator::AnnotationEntry;
using c10::CachingDeviceAllocator::BlockInfo;
using c10::CachingDeviceAllocator::CreateContextFn;
using c10::CachingDeviceAllocator::DeviceStats;
using c10::CachingDeviceAllocator::RecordContext;
using c10::CachingDeviceAllocator::SegmentInfo;
using c10::CachingDeviceAllocator::TraceEntry;

struct AllocatorState {
  virtual ~AllocatorState() = default;
};

struct AllocatorConfigInfo {
  double garbage_collection_threshold;
  size_t max_split_size;
  size_t pinned_num_register_threads;
  bool expandable_segments;
  bool release_lock_on_malloc;
  bool pinned_use_host_register;
  bool graph_capture_record_stream_reuse;
  std::string last_allocator_settings;
  std::vector<size_t> roundup_power2_divisions;
};

struct SnapshotInfo {
  std::vector<SegmentInfo> segments;
  std::vector<std::vector<TraceEntry>> device_traces;
  std::vector<AnnotationEntry> external_annotations;
  AllocatorConfigInfo config_metadata;
};

// returns the pointers freed in the pool
// and the pointers allocated. Note: a pointer
// may appear in both freed and allocated
struct CheckpointDelta {
  std::vector<void*> ptrs_freed;
  std::vector<at::DataPtr> dataptrs_allocd;
};

using OutOfMemoryObserver = std::function<void(
    int64_t device,
    size_t allocated,
    size_t device_total,
    size_t device_free)>;

struct ShareableHandle {
  ptrdiff_t offset;
  std::string handle;
};

struct StreamSegmentSize {
  StreamSegmentSize(musaStream_t s, bool small, size_t sz)
      : stream(s), is_small_pool(small), total_size(sz) {}
  musaStream_t stream;
  bool is_small_pool;
  size_t total_size;
};

class MUSAAllocator : public DeviceAllocator {
 public:
  virtual void* raw_alloc(size_t nbytes) = 0;
  virtual void* raw_alloc_with_stream(size_t nbytes, musaStream_t stream) = 0;
  virtual void raw_delete(void* ptr) = 0;
  virtual void init(int device_count) = 0;
  virtual double getMemoryFraction(c10::DeviceIndex device) = 0;
  virtual void setMemoryFraction(double fraction, c10::DeviceIndex device) = 0;
  virtual std::vector<StreamSegmentSize> getExpandableSegmentSizes(
      c10::DeviceIndex device) = 0;
  virtual void enable(bool value) = 0;
  virtual bool isEnabled() const = 0;
  virtual void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) = 0;
  virtual void* getBaseAllocation(void* ptr, size_t* size) = 0;
  // Keep for BC only
  virtual void recordStream(const DataPtr& ptr, MUSAStream stream) = 0;
  void recordStream(const DataPtr& ptr, c10::Stream stream) override {
    MUSAStream musa_stream = MUSAStream(stream);
    recordStream(ptr, musa_stream);
  }
  virtual SnapshotInfo snapshot(
      MempoolId_t mempool_id = {0, 0},
      bool include_traces = true) = 0;
  virtual void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(musaStream_t)> filter) = 0;
  virtual void endAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id) = 0;
  virtual void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) = 0;
  virtual int getPoolUseCount(
      c10::DeviceIndex /*device*/,
      MempoolId_t /*mempool_id*/) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support getPoolUseCount. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void createOrIncrefPool(
      c10::DeviceIndex /*device*/,
      MempoolId_t /*mempool_id*/,
      std::shared_ptr<MUSAAllocator> allocator = nullptr) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support createOrIncrefPool. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void setUseOnOOM(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      bool use_on_oom) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support setUseOnOOM. "
        "If you need it, please file an issue describing your use case.");
  }

  virtual void setNoSplit(c10::DeviceIndex device, MempoolId_t mempool_id) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support setNoSplit. "
        "If you need it, please file an issue describing your use case.");
  }

  // returns true if the allocated blocks are equal to expected live allocations
  virtual bool checkPoolLiveAllocations(
      c10::DeviceIndex /*device*/,
      MempoolId_t /*mempool_id*/,
      const std::unordered_set<void*>& /*expected_live_allocations*/) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support checkPoolLiveAllocations. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual ShareableHandle shareIpcHandle(void* ptr) = 0;
  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) = 0;
  virtual bool isHistoryEnabled() {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support recordHistory. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when,
      bool clearHistory,
      const std::vector<std::string>& skip_actions) = 0;
  virtual void recordAnnotation(
      const std::vector<std::pair<std::string, std::string>>& /*md*/) {}
  virtual void pushCompileContext(std::string& md) {}
  virtual void popCompileContext() {}
  virtual void setUserMetadata(const std::string& metadata) {}
  virtual std::string getUserMetadata() {
    return "";
  }
  virtual void attachOutOfMemoryObserver(OutOfMemoryObserver observer) = 0;

  // Attached AllocatorTraceTracker callbacks will be called while the
  // per-device allocator lock is held. Any additional locks taken from within
  // the callback must be proven to always have the lock order that never
  // triggers a deadlock. In particular, Python's GIL may be held when
  // calling the allocator so it is unsafe to try to acquire the GIL in this
  // callback.
  virtual void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) = 0;

  virtual void enablePeerAccess(
      c10::DeviceIndex dev,
      c10::DeviceIndex dev_to_access) = 0;

  // memory not allocated from musaMalloc cannot be copied
  // across devices using musaMemcpyAsync if peer to peer access is disabled.
  // instead it requires musaMemcpyAsyncPeer
  //  with P2P Enabled, all combinations work
  //  with P2P Disabled:
  //                       musaMalloc musaMallocAsync/muMemMap
  // musaMemcpyAsyncPeer   works      works
  // musaMemcpyAsync       works      error

  // This function performs chooses to use the Peer version of
  // memcpy if required based on where the allocated put dst/src.
  virtual musaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      musaStream_t stream,
      bool p2p_enabled) = 0;
  virtual std::shared_ptr<AllocatorState> getCheckpointState(
      c10::DeviceIndex device,
      MempoolId_t id) = 0;
  virtual CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<AllocatorState> pps) = 0;
  virtual std::string name() = 0;
  std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) override {
    c10::DeviceGuard device_guard({at::kMUSA, device});
    size_t free = 0;
    size_t total = 0;
    C10_MUSA_CHECK(musaMemGetInfo(&free, &total));
    return {free, total};
  }
};

C10_MUSA_API extern std::atomic<MUSAAllocator*> allocator;

inline MUSAAllocator* get() {
  return allocator.load();
}

inline void* raw_alloc(size_t nbytes) {
  return get()->raw_alloc(nbytes);
}

inline void* raw_alloc_with_stream(size_t nbytes, musaStream_t stream) {
  return get()->raw_alloc_with_stream(nbytes, stream);
}

inline void raw_delete(void* ptr) {
  return get()->raw_delete(ptr);
}

inline void init(int device_count) {
  return get()->init(device_count);
}

inline double getMemoryFraction(c10::DeviceIndex device) {
  return get()->getMemoryFraction(device);
}

inline void setMemoryFraction(double fraction, c10::DeviceIndex device) {
  return get()->setMemoryFraction(fraction, device);
}

inline std::vector<StreamSegmentSize> getExpandableSegmentSizes(
    c10::DeviceIndex device) {
  return get()->getExpandableSegmentSizes(device);
}

inline void emptyCache(MempoolId_t mempool_id = {0, 0}) {
  return get()->emptyCache(mempool_id);
}

inline void enable(bool value) {
  return get()->enable(value);
}

inline bool isEnabled() {
  return get()->isEnabled();
}

inline void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) {
  return get()->cacheInfo(device, largestBlock);
}

inline void* getBaseAllocation(void* ptr, size_t* size) {
  return get()->getBaseAllocation(ptr, size);
}

inline void recordStream(const DataPtr& dataPtr, MUSAStream stream) {
  return get()->recordStream(dataPtr, stream);
}

inline c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
    c10::DeviceIndex device) {
  return get()->getDeviceStats(device);
}

inline void resetAccumulatedStats(c10::DeviceIndex device) {
  return get()->resetAccumulatedStats(device);
}

inline void resetPeakStats(c10::DeviceIndex device) {
  return get()->resetPeakStats(device);
}

inline SnapshotInfo snapshot(
    MempoolId_t mempool_id = {0, 0},
    bool include_traces = true) {
  return get()->snapshot(mempool_id, include_traces);
}

inline std::shared_ptr<AllocatorState> getCheckpointState(
    c10::DeviceIndex device,
    MempoolId_t id) {
  return get()->getCheckpointState(device, id);
}

inline CheckpointDelta setCheckpointPoolState(
    c10::DeviceIndex device,
    std::shared_ptr<AllocatorState> pps) {
  return get()->setCheckpointPoolState(device, std::move(pps));
}

// MUSAGraph interactions
inline void beginAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::function<bool(musaStream_t)> filter) {
  get()->beginAllocateToPool(device, mempool_id, std::move(filter));
}

inline void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  get()->endAllocateToPool(device, mempool_id);
}

inline void recordHistory(
    bool enabled,
    CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    RecordContext when,
    bool clearHistory,
    const std::vector<std::string>& skip_actions = {}) {
  return get()->recordHistory(
      enabled,
      context_recorder,
      alloc_trace_max_entries,
      when,
      clearHistory,
      skip_actions);
}

inline void recordAnnotation(
    const std::vector<std::pair<std::string, std::string>>& md) {
  return get()->recordAnnotation(md);
}

inline void pushCompileContext(std::string& md) {
  return get()->pushCompileContext(md);
}

inline void popCompileContext() {
  return get()->popCompileContext();
}

inline bool isHistoryEnabled() {
  return get()->isHistoryEnabled();
}

inline bool checkPoolLiveAllocations(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    const std::unordered_set<void*>& expected_live_allocations) {
  return get()->checkPoolLiveAllocations(
      device, mempool_id, expected_live_allocations);
}

inline void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
  return get()->attachOutOfMemoryObserver(std::move(observer));
}

inline void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) {
  return get()->attachAllocatorTraceTracker(std::move(tracker));
}

inline void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return get()->releasePool(device, mempool_id);
}
inline void createOrIncrefPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::shared_ptr<MUSAAllocator> allocator = nullptr) {
  get()->createOrIncrefPool(device, mempool_id, std::move(allocator));
}
inline void setUseOnOOM(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    bool use_on_oom = true) {
  get()->setUseOnOOM(device, mempool_id, use_on_oom);
}

inline void setNoSplit(c10::DeviceIndex device, MempoolId_t mempool_id) {
  get()->setNoSplit(device, mempool_id);
}

inline int getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return get()->getPoolUseCount(device, mempool_id);
}

// Not part of MUSA_ALLOCATOR_BACKEND_INTERFACE
inline std::shared_ptr<void> getIpcDevPtr(std::string handle) {
  return get()->getIpcDevPtr(std::move(handle));
}

inline ShareableHandle shareIpcHandle(void* ptr) {
  return get()->shareIpcHandle(ptr);
}

inline std::string name() {
  return get()->name();
}

inline musaError_t memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    musaStream_t stream,
    bool p2p_enabled) {
  return get()->memcpyAsync(
      dst, dstDevice, src, srcDevice, count, stream, p2p_enabled);
}

inline void enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access) {
  return get()->enablePeerAccess(dev, dev_to_access);
}

inline void setUserMetadata(const std::string& metadata) {
  get()->setUserMetadata(metadata);
}

inline std::string getUserMetadata() {
  return get()->getUserMetadata();
}

} // namespace c10::musa::MUSACachingAllocator

namespace c10::musa {

struct C10_MUSA_API MemPool {
  MemPool(
      std::shared_ptr<MUSACachingAllocator::MUSAAllocator> allocator = nullptr,
      bool is_user_created = true,
      bool use_on_oom = false,
      bool no_split = false);
  MemPool(const MemPool&) = delete;
  MemPool(MemPool&&) = default;
  MemPool& operator=(const MemPool&) = delete;
  MemPool& operator=(MemPool&&) = default;
  ~MemPool();

  MempoolId_t id();
  int use_count();
  c10::DeviceIndex device();
  static MempoolId_t graph_pool_handle(bool is_user_created = true);

 private:
  static std::atomic<CaptureId_t> uid_;
  static std::atomic<CaptureId_t> uuid_;
  bool is_user_created_;
  MempoolId_t id_;
  c10::DeviceIndex device_;
};

} // namespace c10::musa
