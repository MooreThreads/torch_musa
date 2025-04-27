#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/StorageImpl.h>
#include <c10/musa/MUSA_PORT_Macros.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/Registry.h>

#include "torch_musa/csrc/core/MUSAGraphsC10Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <array>
#include <atomic>
#include <mutex>
#include <set>
#include <unordered_set>

namespace c10 {

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
// class C10_MUSA_API FreeMemoryCallback {
class C10_MUSA_API FreeMemoryCallback {
 public:
  virtual ~FreeMemoryCallback() = default;
  virtual bool Execute() = 0;
};

C10_DECLARE_REGISTRY(FreeMusaMemoryCallbacksRegistry, FreeMemoryCallback);
#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(FreeMusaMemoryCallbacksRegistry, name, __VA_ARGS__);

namespace musa {

// TODO: Turn this into an honest to goodness class. I briefly attempted to do
// this, but it was a bit irritating to figure out how to also correctly
// apply pimpl pattern so I didn't have to leak any internal implementation
// details in the header (MUSACachingAllocator could be made a pimpl, but
// you also need to appropriately define a class which is a subclass
// of Allocator. Not impossible, but required a bit more surgery than
// I wanted to do at the time.)
//
// Why is this using a namespace rather than old-style THCCachingAllocator_
// prefix?  Mostly because it made the HIPify rules easier to write; _ is
// not counted as a word boundary, so you would otherwise have to list each
// of these functions.

namespace MUSACachingAllocator {

extern const size_t kLargeBuffer;

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

  // SUM: bytes allocated by this memory alocator
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;
  // SUM: bytes requested by client code
  StatArray requested_bytes;

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

typedef std::shared_ptr<GatheredContext> (*CreateContextFn)(void);

// Struct containing info of an allocation block (i.e. a fractional part of a
// musaMalloc)..
struct BlockInfo {
  int64_t size = 0;
  int64_t requested_size = 0;
  int32_t gc_counter = 0;
  bool allocated = false;
  bool active = false;
  std::shared_ptr<GatheredContext>
      context_when_allocated; // per-watcher context
};

// Struct containing info of a memory segment (i.e. one contiguous musaMalloc).
struct SegmentInfo {
  int64_t device = 0;
  int64_t address = 0;
  int64_t total_size = 0;
  int64_t requested_size = 0; // unrounded, actually requested size
  int64_t allocated_size = 0;
  int64_t active_size = 0;
  musaStream_t stream = 0;
  bool is_large = false;
  bool is_expandable = false;
  MempoolId_t owner_private_pool_id = {0, 0}; // used by musaGraph
  std::vector<BlockInfo> blocks;
  std::shared_ptr<GatheredContext> context_when_allocated;
};

struct AllocatorState {
  virtual ~AllocatorState() = default;
};

union trace_time_ {
  time_t t_;
  approx_time_t approx_t_;
};

struct TraceEntry {
  enum Action {
    ALLOC, // API made to the caching allocator for new memory
    FREE_REQUESTED, // API call made to the caching allocator to free memory
    FREE_COMPLETED, // The allocator might have to delay a free because
                    // it is still in use on another stream via record_stream
                    // This event is generated when a free actually completes.
    SEGMENT_ALLOC, // a call to musaMalloc to get more memory from the OS
    SEGMENT_FREE, // a call to musaFree to return memory to the OS (e.g. to
                  // defragment or empty_caches)
    SEGMENT_MAP, // a call to muMemMap (used with expandable_segments)
    SEGMENT_UNMAP, // unmap part of a segment (used with expandable segments)
    SNAPSHOT, // a call to snapshot, used to correlate memory snapshots to trace
              // events
    OOM // the allocator threw an OutOfMemoryError (addr_ is the amount of free
        // bytes reported by musa)
  };
  TraceEntry(
      Action action,
      int device,
      int64_t addr,
      size_t size,
      musaStream_t stream,
      approx_time_t time,
      std::shared_ptr<GatheredContext> context = nullptr)
      : action_(action),
        device_(device),
        addr_(addr),
        context_(std::move(context)),
        stream_(stream),
        size_(size) {
    time_.approx_t_ = time;
  }
  Action action_;
  int device_;
  int64_t addr_; // for OOM, this is the amount of free bytes reported by musa
  std::shared_ptr<GatheredContext> context_;
  musaStream_t stream_;
  int64_t size_;
  trace_time_ time_;
};

struct SnapshotInfo {
  std::vector<SegmentInfo> segments;
  std::vector<std::vector<TraceEntry>> device_traces;
};

// returns the pointers freed in the pool
// and the pointers allocated. Note: a pointer
// may appear in both freed and allocated
struct CheckpointDelta {
  std::vector<void*> ptrs_freed;
  std::vector<at::DataPtr> dataptrs_allocd;
};

enum struct RecordContext {
  NEVER = 0,
  STATE = 1, // only keep stacks for active allocations
  ALLOC = 2, // additionally keep stacks for allocations in the trace history
  ALL = 3, // additionally record stacks for when something is freed
};

// Size pretty-printer
std::string format_size(uint64_t size);

using OutOfMemoryObserver = std::function<void(
    int64_t device,
    int64_t allocated,
    int64_t device_total,
    int64_t device_free)>;

using AllocatorTraceTracker = std::function<void(const TraceEntry&)>;

class MUSAAllocator : public Allocator {
 public:
  virtual void* raw_alloc(size_t nbytes) = 0;
  virtual void* raw_alloc_with_stream(size_t nbytes, musaStream_t stream) = 0;
  virtual void raw_delete(void* ptr) = 0;
  virtual void init(int device_count) = 0;
  virtual bool initialized() = 0;
  virtual void setMemoryFraction(double fraction, int device) = 0;
  virtual void emptyCache() = 0;
  virtual void cacheInfo(int dev_id, size_t* largestBlock) = 0;
  virtual void* getBaseAllocation(void* ptr, size_t* size) = 0;
  virtual void recordStream(const DataPtr&, MUSAStream stream) = 0;
  virtual DeviceStats getDeviceStats(int device) = 0;
  virtual void resetAccumulatedStats(int device) = 0;
  virtual void resetPeakStats(int device) = 0;
  virtual SnapshotInfo snapshot() = 0;
  virtual void beginAllocateStreamToPool(
      int device,
      musaStream_t stream,
      MempoolId_t mempool_id) = 0;
  virtual void endAllocateStreamToPool(int device, musaStream_t stream) = 0;
  virtual void releasePool(int device, MempoolId_t mempool_id) = 0;
  // returns true if the allocated blocks are equal to expected live allocations
  virtual bool checkPoolLiveAllocations(
      int device,
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support checkPoolLiveAllocations. "
        "If you need it, please file an issue describing your use case.");
  }
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
      RecordContext when) = 0;
  virtual void attachOutOfMemoryObserver(OutOfMemoryObserver observer) = 0;

  // Attached AllocatorTraceTracker callbacks will be called while the
  // per-device allocator lock is held. Any additional locks taken from within
  // the callback must be proven to always have the lock order that never
  // triggers a deadlock. In particular, Python's GIL may be held when
  // calling the allocator so it is unsafe to try to acquire the GIL in this
  // callback.
  virtual void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) = 0;

  virtual void enablePeerAccess(int dev, int dev_to_access) = 0;

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
      int device,
      MempoolId_t id) = 0;
  virtual CheckpointDelta setCheckpointPoolState(
      int device,
      std::shared_ptr<AllocatorState> pps) = 0;
  virtual std::string name() = 0;
};

// Allocator object, statically initialized
// See BackendInitializer in MUSACachingAllocator.cpp.
// Atomic loads on x86 are just normal loads,
// (atomic stores are different), so reading this value
// is no different than loading a pointer.
C10_MUSA_API extern std::atomic<MUSAAllocator*> allocator;

inline MUSAAllocator* get() {
  return allocator.load();
}

// Called directly by clients.
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

inline void setMemoryFraction(double fraction, int device) {
  return get()->setMemoryFraction(fraction, device);
}

inline void emptyCache() {
  return get()->emptyCache();
}

inline void cacheInfo(int dev_id, size_t* largestBlock) {
  return get()->cacheInfo(dev_id, largestBlock);
}

inline void* getBaseAllocation(void* ptr, size_t* size) {
  return get()->getBaseAllocation(ptr, size);
}

inline void recordStream(const DataPtr& dataPtr, MUSAStream stream) {
  return get()->recordStream(dataPtr, stream);
}

inline DeviceStats getDeviceStats(int device) {
  return get()->getDeviceStats(device);
}

inline void resetAccumulatedStats(int device) {
  return get()->resetAccumulatedStats(device);
}

inline void resetPeakStats(int device) {
  return get()->resetPeakStats(device);
}

inline SnapshotInfo snapshot() {
  return get()->snapshot();
}

inline std::shared_ptr<AllocatorState> getCheckpointState(
    int device,
    MempoolId_t id) {
  return get()->getCheckpointState(device, id);
}

inline CheckpointDelta setCheckpointPoolState(
    int device,
    std::shared_ptr<AllocatorState> pps) {
  return get()->setCheckpointPoolState(device, pps);
}

// MUSAGraph interactions
inline void beginAllocateStreamToPool(
    int device,
    musaStream_t stream,
    MempoolId_t mempool_id) {
  return get()->beginAllocateStreamToPool(device, stream, mempool_id);
}

inline void endAllocateStreamToPool(int device, musaStream_t stream) {
  return get()->endAllocateStreamToPool(device, stream);
}

inline void recordHistory(
    bool enabled,
    CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    RecordContext when) {
  return get()->recordHistory(
      enabled, context_recorder, alloc_trace_max_entries, when);
}

inline bool isHistoryEnabled() {
  return get()->isHistoryEnabled();
}

inline bool checkPoolLiveAllocations(
    int device,
    MempoolId_t mempool_id,
    const std::unordered_set<void*>& expected_live_allocations) {
  return get()->checkPoolLiveAllocations(
      device, mempool_id, expected_live_allocations);
}

inline void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
  return get()->attachOutOfMemoryObserver(observer);
}

inline void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) {
  return get()->attachAllocatorTraceTracker(tracker);
}

inline void releasePool(int device, MempoolId_t mempool_id) {
  return get()->releasePool(device, mempool_id);
}
// Not part of MUSA_ALLOCATOR_BACKEND_INTERFACE
inline std::shared_ptr<void> getIpcDevPtr(std::string handle) {
  return get()->getIpcDevPtr(handle);
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

inline void enablePeerAccess(int dev, int dev_to_access) {
  return get()->enablePeerAccess(dev, dev_to_access);
}

} // namespace MUSACachingAllocator
} // namespace musa
} // namespace c10
