#pragma once

#include <c10/core/Allocator.h>
// #include <c10/musa/MUSAMacros.h>

#include "torch_musa/csrc/core/MUSACachingAllocator.h"
#include "torch_musa/csrc/core/MUSAGraphsC10Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <mutex>

namespace torch::musa::MUSAPluggableAllocator {

using MallocFuncType = void*(size_t, int, musaStream_t);
using FreeFuncType = void(void*, size_t, int, musaStream_t);

// See Note [How UniqueVoidPtr is implemented] and look at COWDeleter'
// implementation to know how MUSAPluggableAllocatorDeleterContext works A
// MUSAPluggableAllocatorDeleterContext object is used as the `ctx` argument for
// DataPtr. We need context because a user can use multiple allocators in the
// same PyTorch program, and the allocators can have different free functions,
// such as: free, musaFree, musaFreeAsync, mcclMemFree etc.
struct MUSAPluggableAllocatorDeleterContext {
  explicit MUSAPluggableAllocatorDeleterContext(
      std::function<FreeFuncType> free_fn,
      void* data,
      size_t size,
      int device,
      musaStream_t stream);

  void free();

 private:
  std::function<FreeFuncType> free_fn_;
  void* data_;
  size_t size_;
  int device_;
  musaStream_t stream_;
};

using streamType = c10::musa::MUSAStream;
using c10::musa::MUSACachingAllocator::MUSAAllocator;

std::shared_ptr<MUSAAllocator> getCurrentAllocator();
std::shared_ptr<MUSAAllocator> createCustomAllocator(
    std::function<MallocFuncType> alloc_fn,
    std::function<FreeFuncType> free_fn);
void changeCurrentAllocator(const std::shared_ptr<MUSAAllocator>& allocator);

struct _AllocationMetadata {
  _AllocationMetadata();
  _AllocationMetadata(
      size_t size,
      c10::DeviceIndex device_idx,
      musaStream_t stream);
  size_t size;
  c10::DeviceIndex device_idx;
  musaStream_t stream{};
};

struct MUSAPluggableAllocator : public MUSAAllocator {
  MUSAPluggableAllocator(
      std::function<MallocFuncType> alloc_fn,
      std::function<FreeFuncType> free_fn);

  MUSAPluggableAllocator(MUSAPluggableAllocator& other);
  MUSAPluggableAllocator(MUSAPluggableAllocator&& other) = delete;
  MUSAPluggableAllocator& operator=(MUSAPluggableAllocator& other) = delete;
  MUSAPluggableAllocator& operator=(MUSAPluggableAllocator&& other) = delete;
  ~MUSAPluggableAllocator() override = default;

  void set_init_fn(std::function<void(int)> init_fn);

  void set_reset_fn(std::function<void()> reset_fn);

  void set_memory_fraction_fn(
      std::function<void(double, int)> memory_fraction_fn);

  void set_base_alloc_fn(std::function<void*(void*, size_t*)> base_alloc_fn);

  void set_record_stream_fn(
      std::function<void(void* ptr, musaStream_t stream)> record_stream_fn);

  void set_begin_allocate_to_pool_fn(
      std::function<
          void(int, c10::musa::MempoolId_t, std::function<bool(musaStream_t)>)>
          capture_begin_fn);

  void set_end_allocate_to_pool_fn(
      std::function<void(int, c10::musa::MempoolId_t)> capture_about_to_end_fn);

  void set_release_pool_fn(
      std::function<void(int, c10::musa::MempoolId_t)> capture_destroy_fn);

  void* malloc(size_t size, c10::DeviceIndex device, musaStream_t stream);

  c10::DataPtr allocate(size_t size) override;
  c10::DeleterFnPtr raw_deleter() const override;

  void* raw_alloc(size_t nbytes) override;
  void* raw_alloc_with_stream(size_t nbytes, musaStream_t stream) override;
  void raw_delete(void* ptr) override;
  void init(int device_count) override;
  bool initialized() override;
  void setMemoryFraction(double fraction, c10::DeviceIndex device) override;
  void emptyCache() override;
  void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) override;
  void* getBaseAllocation(void* ptr, size_t* size) override;

  void recordStream(const c10::DataPtr&, streamType stream) override;

  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override;
  void resetAccumulatedStats(c10::DeviceIndex device) override;
  void resetPeakStats(c10::DeviceIndex device) override;
  c10::musa::MUSACachingAllocator::SnapshotInfo snapshot() override;
  void beginAllocateToPool(
      c10::DeviceIndex device,
      c10::musa::MempoolId_t mempool_id,
      std::function<bool(musaStream_t)>) override;
  void endAllocateToPool(
      c10::DeviceIndex device,
      c10::musa::MempoolId_t mempool_id) override;
  void releasePool(c10::DeviceIndex device, c10::musa::MempoolId_t mempool_id)
      override;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override;
  c10::musa::MUSACachingAllocator::ShareableHandle shareIpcHandle(
      void*) override;
  void recordHistory(
      bool enabled,
      c10::musa::MUSACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      c10::musa::MUSACachingAllocator::RecordContext when) override;
  void attachOutOfMemoryObserver(
      c10::musa::MUSACachingAllocator::OutOfMemoryObserver observer) override;
  void attachAllocatorTraceTracker(
      c10::musa::MUSACachingAllocator::AllocatorTraceTracker tracker) override;
  std::shared_ptr<c10::musa::MUSACachingAllocator::AllocatorState>
  getCheckpointState(c10::DeviceIndex device, at::musa::MempoolId_t id)
      override;
  c10::musa::MUSACachingAllocator::CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<c10::musa::MUSACachingAllocator::AllocatorState> pps)
      override;
  void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access)
      override;
  musaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      musaStream_t stream,
      bool p2p_enabled) override;
  std::string name() override;
  void copy_data(void* dest, const void* src, std::size_t count) const final;

 protected:
  std::function<MallocFuncType> alloc_fn_;
  std::function<FreeFuncType> free_fn_;
  std::function<void(int)> init_fn_;
  std::function<void()> reset_fn_;
  std::function<void(double, int)> memory_fraction_fn_;
  std::function<void*(void*, size_t*)> base_alloc_fn_;
  std::function<void(void* ptr, musaStream_t stream)> record_stream_fn_;
  std::function<
      void(int, c10::musa::MempoolId_t, std::function<bool(musaStream_t)>)>
      begin_allocate_to_pool_fn_;
  std::function<void(int, c10::musa::MempoolId_t)> end_allocate_to_pool_fn_;
  std::function<void(int, c10::musa::MempoolId_t)> relase_pool_fn_;
  std::mutex allocator_mutex_;
  // We do the bookeeping here in order to simplify custom allocators
  std::unordered_map<void*, _AllocationMetadata> allocation_metadata_;

  bool initialized_ = false;
};
} // namespace torch::musa::MUSAPluggableAllocator
