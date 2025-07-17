#include <mutex>
#include <unordered_map>
#include <utility>

#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAPluggableAllocator.h"

namespace torch::musa::MUSAPluggableAllocator {

MUSAPluggableAllocatorDeleterContext::MUSAPluggableAllocatorDeleterContext(
    std::function<FreeFuncType> free_fn,
    void* data,
    size_t size,
    int device,
    musaStream_t stream)
    : free_fn_(free_fn),
      data_(data),
      size_(size),
      device_(device),
      stream_(stream) {}

void MUSAPluggableAllocatorDeleterContext::free() {
  free_fn_(data_, size_, device_, stream_);
  delete this;
}

int device_count = 0;

void custom_raw_deleter(void* ptr);

_AllocationMetadata::_AllocationMetadata() : size(0), device_idx(-1) {}

_AllocationMetadata::_AllocationMetadata(
    size_t size,
    c10::DeviceIndex device_idx,
    musaStream_t stream)
    : size(size), device_idx(device_idx), stream(stream) {}

// This is a fast API to just register allocators
// based on function pointers (ie. external .so libraries)
// This avoids having to link against libtorch for C++ based custom allocators
// And also use this from python
MUSAPluggableAllocator::MUSAPluggableAllocator(
    std::function<MallocFuncType> alloc_fn,
    std::function<FreeFuncType> free_fn)
    : alloc_fn_(std::move(alloc_fn)), free_fn_(std::move(free_fn)) {}

MUSAPluggableAllocator::MUSAPluggableAllocator(MUSAPluggableAllocator& other)
    : alloc_fn_(other.alloc_fn_),
      free_fn_(other.free_fn_),
      init_fn_(other.init_fn_),
      reset_fn_(other.reset_fn_),
      memory_fraction_fn_(other.memory_fraction_fn_),
      base_alloc_fn_(other.base_alloc_fn_),
      record_stream_fn_(other.record_stream_fn_),
      begin_allocate_to_pool_fn_(other.begin_allocate_to_pool_fn_),
      end_allocate_to_pool_fn_(other.end_allocate_to_pool_fn_),
      relase_pool_fn_(other.relase_pool_fn_) {}

void MUSAPluggableAllocator::set_init_fn(std::function<void(int)> init_fn) {
  init_fn_ = std::move(init_fn);
}

void MUSAPluggableAllocator::set_reset_fn(std::function<void()> reset_fn) {
  reset_fn_ = std::move(reset_fn);
}

void MUSAPluggableAllocator::set_memory_fraction_fn(
    std::function<void(double, int)> memory_fraction_fn) {
  memory_fraction_fn_ = std::move(memory_fraction_fn);
}

void MUSAPluggableAllocator::set_base_alloc_fn(
    std::function<void*(void*, size_t*)> base_alloc_fn) {
  base_alloc_fn_ = std::move(base_alloc_fn);
}

void MUSAPluggableAllocator::set_record_stream_fn(
    std::function<void(void* ptr, musaStream_t stream)> record_stream_fn) {
  record_stream_fn_ = std::move(record_stream_fn);
}

void MUSAPluggableAllocator::set_begin_allocate_to_pool_fn(
    std::function<
        void(int, c10::musa::MempoolId_t, std::function<bool(musaStream_t)>)>
        capture_begin_fn) {
  begin_allocate_to_pool_fn_ = std::move(capture_begin_fn);
}

void MUSAPluggableAllocator::set_end_allocate_to_pool_fn(
    std::function<void(int, c10::musa::MempoolId_t)> capture_about_to_end_fn) {
  end_allocate_to_pool_fn_ = std::move(capture_about_to_end_fn);
}

void MUSAPluggableAllocator::set_release_pool_fn(
    std::function<void(int, c10::musa::MempoolId_t)> capture_destroy_fn) {
  relase_pool_fn_ = std::move(capture_destroy_fn);
}

void* MUSAPluggableAllocator::malloc(
    size_t size,
    c10::DeviceIndex device,
    musaStream_t stream) {
  void* r = alloc_fn_(size, device, stream);
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    allocation_metadata_.emplace(r, _AllocationMetadata(size, device, stream));
  }
  return r;
}

c10::DataPtr MUSAPluggableAllocator::allocate(size_t size) {
  c10::DeviceIndex device = -1;
  C10_MUSA_CHECK(c10::musa::GetDevice(&device));
  musaStream_t stream = c10::musa::getCurrentMUSAStream(device);
  void* r = this->malloc(size, device, stream);
  auto* ctx = new MUSAPluggableAllocatorDeleterContext(
      free_fn_, r, size, device, stream);
  c10::DataPtr data_ptr = {
      r, ctx, raw_deleter(), c10::Device(c10::DeviceType::PrivateUse1, device)};
  return data_ptr;
}

c10::DeleterFnPtr MUSAPluggableAllocator::raw_deleter() const {
  return &custom_raw_deleter;
}

void* MUSAPluggableAllocator::raw_alloc(size_t nbytes) {
  c10::DeviceIndex device = -1;
  C10_MUSA_CHECK(c10::musa::GetDevice(&device));
  musaStream_t stream = c10::musa::getCurrentMUSAStream(device);
  return malloc(nbytes, device, stream);
}

void* MUSAPluggableAllocator::raw_alloc_with_stream(
    size_t nbytes,
    musaStream_t stream) {
  c10::DeviceIndex device = -1;
  C10_MUSA_CHECK(c10::musa::GetDevice(&device));
  return malloc(nbytes, device, stream);
}

void MUSAPluggableAllocator::raw_delete(void* ptr) {
  musaStream_t stream{};
  c10::DeviceIndex device_idx = -1;
  size_t size = 0;
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    TORCH_CHECK(
        allocation_metadata_.count(ptr),
        "Trying to free a pointer not allocated here");
    _AllocationMetadata& metadata = allocation_metadata_[ptr];
    size = metadata.size;
    device_idx = metadata.device_idx;
    stream = metadata.stream;
    allocation_metadata_.erase(ptr);
  }
  free_fn_(ptr, size, device_idx, stream);
}

void MUSAPluggableAllocator::init(int device_count) {
  if (init_fn_) {
    init_fn_(device_count);
  }
  initialized_ = true;
}

bool MUSAPluggableAllocator::initialized() {
  return initialized_;
}

void MUSAPluggableAllocator::setMemoryFraction(
    double fraction,
    c10::DeviceIndex device) {
  if (memory_fraction_fn_) {
    memory_fraction_fn_(fraction, device);
  }
}

void MUSAPluggableAllocator::emptyCache() {
  if (reset_fn_) {
    return reset_fn_();
  }
}

void MUSAPluggableAllocator::cacheInfo(
    c10::DeviceIndex device,
    size_t* largestBlock) {
  TORCH_CHECK(
      false,
      "MUSAPluggableAllocator does not yet support cacheInfo. "
      "If you need it, please file an issue describing your use case.");
}

void* MUSAPluggableAllocator::getBaseAllocation(void* ptr, size_t* size) {
  if (base_alloc_fn_) {
    return base_alloc_fn_(ptr, size);
  } else {
    return ptr;
  }
}

void MUSAPluggableAllocator::recordStream(
    const c10::DataPtr& ptr,
    streamType stream) {
  if (record_stream_fn_) {
    record_stream_fn_(ptr.get(), stream);
  }
}

c10::CachingDeviceAllocator::DeviceStats MUSAPluggableAllocator::getDeviceStats(
    c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "MUSAPluggableAllocator does not yet support getDeviceStats. "
      "If you need it, please file an issue describing your use case.");
}

void MUSAPluggableAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "MUSAPluggableAllocator does not yet support resetAccumulatedStats. "
      "If you need it, please file an issue describing your use case.");
}

void MUSAPluggableAllocator::resetPeakStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "MUSAPluggableAllocator does not yet support resetPeakStats. "
      "If you need it, please file an issue describing your use case.");
}

c10::musa::MUSACachingAllocator::SnapshotInfo MUSAPluggableAllocator::
    snapshot() {
  TORCH_CHECK(
      false,
      "MUSAPluggableAllocator does not yet support snapshot. "
      "If you need it, please file an issue describing your use case.");
}

c10::musa::MUSACachingAllocator::ShareableHandle MUSAPluggableAllocator::
    shareIpcHandle(void* ptr) {
  TORCH_CHECK(
      false,
      "MUSAPluggableAllocator does not yet support shareIPcHandle. "
      "If you need it, please file an issue describing your use case.");
}

std::shared_ptr<void> MUSAPluggableAllocator::getIpcDevPtr(std::string handle) {
  TORCH_CHECK(
      false,
      "MUSAPluggableAllocator does not yet support getIpcDevPtr. "
      "If you need it, please file an issue describing your use case.");
}

// MUSAGraph interactions
void MUSAPluggableAllocator::beginAllocateToPool(
    c10::DeviceIndex device,
    c10::musa::MempoolId_t mempool_id,
    std::function<bool(musaStream_t)> filter) {
  if (begin_allocate_to_pool_fn_) {
    begin_allocate_to_pool_fn_(device, mempool_id, std::move(filter));
  }
}

void MUSAPluggableAllocator::endAllocateToPool(
    c10::DeviceIndex device,
    c10::musa::MempoolId_t mempool_id) {
  if (end_allocate_to_pool_fn_) {
    end_allocate_to_pool_fn_(device, mempool_id);
  }
}

void MUSAPluggableAllocator::releasePool(
    c10::DeviceIndex device,
    c10::musa::MempoolId_t mempool_id) {
  if (relase_pool_fn_) {
    relase_pool_fn_(device, mempool_id);
  }
}

void MUSAPluggableAllocator::recordHistory(
    bool enabled,
    c10::musa::MUSACachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    c10::musa::MUSACachingAllocator::RecordContext when) {
  TORCH_CHECK(
      false,
      "MUSAPluggableAllocator does not yet support recordHistory. "
      "If you need it, please file an issue describing your use case.");
}

void MUSAPluggableAllocator::attachOutOfMemoryObserver(
    c10::musa::MUSACachingAllocator::OutOfMemoryObserver observer) {
  TORCH_CHECK(
      false,
      "MUSAPluggableAllocator does not yet support attachOutOfMemoryObserver. "
      "If you need it, please file an issue describing your use case.");
}

void MUSAPluggableAllocator::attachAllocatorTraceTracker(
    c10::musa::MUSACachingAllocator::AllocatorTraceTracker tracker) {
  TORCH_CHECK(
      false,
      "MUSAPluggableAllocator does not support attachAllocatorTraceTracker. "
      "attachAllocatorTraceTracker is only used inside Pytorch.");
}

std::shared_ptr<c10::musa::MUSACachingAllocator::AllocatorState>
MUSAPluggableAllocator::getCheckpointState(
    c10::DeviceIndex device,
    at::musa::MempoolId_t id) {
  TORCH_CHECK(
      false,
      "MUSAPluggableAllocator does not yet support getCheckpointState. "
      "If you need it, please file an issue describing your use case.");
}

c10::musa::MUSACachingAllocator::CheckpointDelta MUSAPluggableAllocator::
    setCheckpointPoolState(
        c10::DeviceIndex device,
        std::shared_ptr<c10::musa::MUSACachingAllocator::AllocatorState> pps) {
  TORCH_CHECK(
      false,
      "MUSAPluggableAllocator does not yet support setCheckpointPoolState. "
      "If you need it, please file an issue describing your use case.");
}

void MUSAPluggableAllocator::enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access) {
  c10::musa::MUSAGuard device_guard(dev);
  musaError_t err = musaDeviceEnablePeerAccess(dev_to_access, 0);
  if (err == musaErrorPeerAccessAlreadyEnabled) {
    // ignore and clear the error if access was already enabled
    (void)musaGetLastError();
  } else {
    C10_MUSA_CHECK(err);
  }
}

musaError_t MUSAPluggableAllocator::memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    musaStream_t stream,
    bool p2p_enabled) {
  return musaMemcpyAsync(dst, src, count, musaMemcpyDeviceToDevice, stream);
}

std::string MUSAPluggableAllocator::name() {
  return "pluggable";
}

void MUSAPluggableAllocator::copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  C10_MUSA_CHECK(
      musaMemcpy(dest, src, count, musaMemcpyKind::musaMemcpyDeviceToDevice));
}

std::shared_ptr<c10::musa::MUSACachingAllocator::MUSAAllocator>
    current_custom_allocator;

std::shared_ptr<c10::musa::MUSACachingAllocator::MUSAAllocator>
getCurrentAllocator() {
  return current_custom_allocator;
}

// TODO: add more functions in the argument
std::shared_ptr<c10::musa::MUSACachingAllocator::MUSAAllocator>
createCustomAllocator(
    std::function<MallocFuncType> alloc_fn,
    std::function<FreeFuncType> free_fn) {
  std::shared_ptr<MUSAPluggableAllocator> allocator(
      new MUSAPluggableAllocator(std::move(alloc_fn), std::move(free_fn)));
  allocator->init(device_count);
  return allocator;
}

void changeCurrentAllocator(
    const std::shared_ptr<c10::musa::MUSACachingAllocator::MUSAAllocator>&
        allocator) {
  TORCH_CHECK(
      !c10::musa::MUSACachingAllocator::allocator.load()->initialized(),
      "Can't swap an already initialized allocator");
  c10::musa::MUSACachingAllocator::allocator.store(allocator.get());
  current_custom_allocator = allocator;
}

void custom_raw_deleter(void* ctx) {
  reinterpret_cast<MUSAPluggableAllocatorDeleterContext*>(ctx)->free();
}

} // namespace torch::musa::MUSAPluggableAllocator
