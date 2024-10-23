#ifndef TORCH_MUSA_CSRC_CORE_ALLOCATOR_H_
#define TORCH_MUSA_CSRC_CORE_ALLOCATOR_H_

#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/DeviceType.h>
#include <c10/musa/MUSA_PORT_Macros.h>
#include <list>
#include <set>

#include <mudnn.h>
#include <musa_runtime.h>

#include "torch_musa/csrc/core/MUSAGraphsC10Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace c10 {

class C10_MUSA_API MUSAOutOfMemoryError : public c10::Error {
  using Error::Error;
};

namespace musa {

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

namespace MUSACachingAllocator {

void* raw_alloc(size_t nbytes);
void* raw_alloc_with_stream(size_t nbytes, MUSAStream stream);
void raw_delete(void* ptr);

Allocator* get();

// Size pretty-printer
std::string format_size(uint64_t size);

struct Context {
  virtual ~Context() = default;
};

typedef std::shared_ptr<Context> (*CreateContextFn)(void);

using OutOfMemoryObserver = std::function<void(
    int64_t device,
    int64_t allocated,
    int64_t device_total,
    int64_t device_free)>;

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

struct History {
  void* addr;
  size_t real_size; // unrounded, actually requested size
  std::shared_ptr<Context> context; // per-watcher context
};

// Struct containing info of an allocation block (i.e. a fractional part of a
// musaMalloc)..
struct BlockInfo {
  int64_t size = 0;
  int64_t requested_size = 0;
  int32_t gc_counter = 0;
  bool allocated = false;
  bool active = false;
  std::vector<History> history;
};

// Struct containing info of a memory segment (i.e. one contiguous musaMalloc).
struct SegmentInfo {
  int64_t device = 0;
  int64_t address = 0;
  int64_t total_size = 0;
  int64_t requested_size = 0;
  int64_t allocated_size = 0;
  int64_t active_size = 0;
  musaStream_t stream = 0;
  bool is_large = false;
  std::vector<BlockInfo> blocks;
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
                  // defragement or empty_caches)
    SNAPSHOT, // a call to snapshot, used to correlate memory snapshots to trace
              // events
    OOM // the allocator threw an OutOfMemoryError (addr_ is the amount of free
        // bytes reported by cuda)
  };
  TraceEntry(
      Action action,
      int64_t addr,
      size_t size,
      musaStream_t stream,
      std::shared_ptr<Context> context = nullptr)
      : action_(action),
        addr_(addr),
        context_(context),
        stream_(stream),
        size_(size) {}
  Action action_;
  int64_t addr_; // for OOM, this is the amount of free bytes reported by cuda
  std::shared_ptr<Context> context_;
  musaStream_t stream_;
  int64_t size_;
};

struct SnapshotInfo {
  std::vector<SegmentInfo> segments;
  std::vector<std::vector<TraceEntry>> device_traces;
};

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
  virtual void cacheInfoWithTotal(
      int dev_id,
      size_t* largestBlock,
      size_t* total) = 0;
  virtual void* getBaseAllocation(void* ptr, size_t* size) = 0;
  virtual void recordStream(const DataPtr&, MUSAStream stream) = 0;
  virtual DeviceStats getDeviceStats(int device) = 0;
  virtual void resetAccumulatedStats(int device) = 0;
  virtual void resetPeakStats(int device) = 0;
  virtual SnapshotInfo snapshot() = 0;
  virtual void notifyCaptureBegin(
      int device,
      CaptureId_t graph_id,
      MempoolId_t mempool_id) = 0;
  virtual void notifyCaptureAboutToEnd(int device, CaptureId_t graph_id) = 0;
  virtual void notifyCaptureEnded(int device, CaptureId_t graph_id) = 0;
  virtual void notifyCaptureDestroy(int device, MempoolId_t mempool_id) = 0;
  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) = 0;
  virtual void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      bool alloc_trace_record_context) = 0;
  virtual void attachOutOfMemoryObserver(OutOfMemoryObserver observer) = 0;
  virtual bool needsPoolSpecificPeerAccess() = 0;
  virtual std::string name() = 0;
};

void init(int device_count);
void SetMemoryFraction(double fraction, int device);
void EmptyCache();
void ResetPeakStats();
void ResetPeakStats(int64_t device);
DeviceStats GetDeviceStats(int64_t device);
SnapshotInfo GetMemorySnapshot();
void recordStream(const DataPtr& dataPtr, MUSAStream stream);
void RecordHistory(
    bool enabled,
    CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    bool alloc_trace_record_context);
void AttachOutOfMemoryObserver(OutOfMemoryObserver observer);
void* GetBaseAllocation(void* ptr, size_t* outSize);
std::shared_ptr<void> GetIpcDevPtr(std::string handle);

} // namespace MUSACachingAllocator
} // namespace musa
} // namespace c10
#endif // TORCH_MUSA_CSRC_CORE_ALLOCATOR_H_
