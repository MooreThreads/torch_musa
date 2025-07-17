#include "torch_musa/csrc/core/MusaIPCTypes.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>

#include <ATen/MapAllocator.h>
#include <c10/util/Logging.h>

#include "torch_musa/csrc/core/MUSAFunctions.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace torch::musa {

void warnProducerTerminatedBeforeSharedTensorsReleased() {
  static bool warned = false;
  if (!warned) {
    LOG(WARNING)
        << "Producer process has been terminated before all shared MUSA tensors released.";
    warned = true;
  }
}

struct MusaIPCGlobalEntities {
  // This class is used as a singleton (see musa_ipc_global_entities)
  // This variable is used to track its lifetime to avoid accessing it
  // after it was destroyed which would lead to segmentation faults
  // Note that a trvial type is used which doesn't suffer from construction
  // and destruction order issues
  static bool alive;

  std::mutex ref_counters_mutex_;
  std::atomic<int64_t> sync_events_used_{0};
  std::map<std::string, std::shared_ptr<MusaIPCRefCountersFile>>
      ref_counters_files_;
  std::shared_ptr<MusaIPCRefCountersFile> next_available_ref_counters_file_;
  MusaIPCSentDataLimbo MusaIPCSentDataLimbo_;
  MusaIPCGlobalEntities() {
    alive = true;
  }
  ~MusaIPCGlobalEntities() {
    MusaIPCSentDataLimbo_.collect();
    safe_clean_current_file();
    if (next_available_ref_counters_file_) {
      warnProducerTerminatedBeforeSharedTensorsReleased();
    }
    alive = false;
  }
  void safe_clean_current_file() {
    std::lock_guard<std::mutex> lock(ref_counters_mutex_);
    if (next_available_ref_counters_file_ &&
        next_available_ref_counters_file_->offsets_in_use() == 0) {
      ref_counters_files_.erase(next_available_ref_counters_file_->handle());
      next_available_ref_counters_file_.reset();
    }
  }
};

bool MusaIPCGlobalEntities::alive = false;
MusaIPCGlobalEntities musa_ipc_global_entities;

MusaIPCSentDataLimbo::~MusaIPCSentDataLimbo() {
  collect();
  if (size() > 0) {
    warnProducerTerminatedBeforeSharedTensorsReleased();
  }
}

bool MusaIPCSentDataLimbo::collect() {
  bool freed_memory = false;
  std::vector<std::unique_ptr<MusaIPCSentData>> reset_blocks;
  { // Begin critical section to modify shared blocks
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    std::vector<std::unique_ptr<MusaIPCSentData>> kept_blocks;
    for (auto& sd : shared_blocks_) {
      if (sd->counter_value() > 0) {
        kept_blocks.push_back(std::move(sd));
      } else {
        freed_memory = true;
        reset_blocks.push_back(std::move(sd));
      }
    }
    shared_blocks_ = std::move(kept_blocks);
  }
  // Need to reset blocks out of the critical section here, otherwise it
  // deadlocks.
  for (auto& sd : reset_blocks) {
    sd.reset();
  }
  return freed_memory;
}

void MusaIPCSentDataLimbo::add(std::unique_ptr<MusaIPCSentData> shared_block) {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  static bool warned = false;
  if (shared_blocks_.size() > MUSA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO &&
      !warned) {
    LOG(WARNING)
        << "Producer process tried to deallocate over "
        << MUSA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO
        << " memory blocks referred by consumer processes. Deallocation might be significantly slowed down. "
        << "We assume it will never going to be the case, but if it is, please file but to https://github.com/pytorch/pytorch";
    warned = true;
  }
  shared_blocks_.push_back(std::move(shared_block));
}

uint64_t MusaIPCSentDataLimbo::size() {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  return shared_blocks_.size();
}

void MusaIPCSentDataDelete(void* ptr) {
  std::unique_ptr<MusaIPCSentData> sent_data(
      static_cast<MusaIPCSentData*>(ptr));
  if (!MusaIPCGlobalEntities::alive) {
    return;
  }
  if (sent_data->counter_value() > 0) {
    musa_ipc_global_entities.MusaIPCSentDataLimbo_.add(std::move(sent_data));
  }
  musa_ipc_global_entities.MusaIPCSentDataLimbo_.collect();
}

void ReturnRefCounter(const std::string& handle, uint64_t offset /* unused */) {
  if (!MusaIPCGlobalEntities::alive) {
    return;
  }
  std::lock_guard<std::mutex> lock(
      musa_ipc_global_entities.ref_counters_mutex_);
  auto& map = musa_ipc_global_entities.ref_counters_files_;
  auto it = map.find(handle);
  if (it != map.end()) {
    it->second->return_offset(offset);
    if (it->second->offsets_in_use() == 0 && !it->second->have_offsets()) {
      map.erase(handle);
    }
  }
}

MusaIPCSentData::MusaIPCSentData(
    std::string handle,
    uint64_t offset,
    uint64_t* counter_ptr,
    at::Device device)
    : handle_(std::move(handle)),
      offset_(offset),
      counter_ptr_(counter_ptr),
      original_ptr_(),
      device_(device) {
  // CUDA/MUSA have the unofficial limit on the number of recorded blocking
  // interprocess events, to prevent using of all events, we are switching to
  // StreamSync before limit reached.
  //
  //  ```python
  //  import torch, torch_musa
  //  a = [ torch.musa.Event(
  //      enable_timing=False, blocking=True, interprocess=True) for i in
  //      range(30000) ]
  //  [i.record() for i in a]
  //  ```
  //
  if (musa_ipc_global_entities.sync_events_used_.load() <
      MUSA_IPC_MAXIMUM_EVENTS_TO_USE) {
    // TODO: More efficient would be to create event inside of main thread (at
    // the moment of the queue.put). The reason this is more efficient is
    // because the main thread may have queued extra work on the stream, which
    // this event will consequently wait for (uselessly).
    musa_ipc_global_entities.sync_events_used_++;
    C10_MUSA_CHECK(musaEventCreateWithFlags(
        &event_,
        musaEventDisableTiming | musaEventInterprocess |
            musaEventBlockingSync));
    C10_MUSA_CHECK(musaEventRecord(
        event_, c10::musa::getCurrentMUSAStream(device.index())));
    event_sync_required_ = true;
  } else {
    auto stream = c10::musa::getCurrentMUSAStream(device.index());
    c10::musa::stream_synchronize(stream);
    event_ = nullptr;
    event_sync_required_ = false;
  }
}

MusaIPCSentData::~MusaIPCSentData() {
  ReturnRefCounter(handle_, offset_);
  try {
    if (event_sync_required_) {
      at::musa::MUSAGuard device_guard(device_.index());
      C10_MUSA_CHECK(musaEventDestroy(event_));
      if (!MusaIPCGlobalEntities::alive) {
        return;
      }
      musa_ipc_global_entities.sync_events_used_--;
    }
  } catch (...) { /* No throw */
  }
}

uint64_t MusaIPCSentData::counter_value() {
  return *counter_ptr_;
}

at::DataPtr GetNewRefCountedSentData(void* data, at::Device device) {
  {
    std::lock_guard<std::mutex> lock(
        musa_ipc_global_entities.ref_counters_mutex_);
    if (!musa_ipc_global_entities.next_available_ref_counters_file_) {
      std::string ref_counter_handle = at::NewProcessWideShmHandle();

      int flags =
          at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
      at::DataPtr sptr = at::RefcountedMapAllocator::makeDataPtr(
          ref_counter_handle.c_str(),
          flags,
          sizeof(int64_t) * MUSA_IPC_REF_COUNTER_FILE_SIZE,
          nullptr);
      auto rc = std::make_shared<MusaIPCRefCountersFile>(
          ref_counter_handle, MUSA_IPC_REF_COUNTER_FILE_SIZE, std::move(sptr));
      musa_ipc_global_entities.ref_counters_files_[ref_counter_handle] = rc;
      musa_ipc_global_entities.next_available_ref_counters_file_ = rc;
    }
  }
  musa_ipc_global_entities.next_available_ref_counters_file_->set_counter(1);
  auto sent_data = new MusaIPCSentData(
      musa_ipc_global_entities.next_available_ref_counters_file_->handle(),
      musa_ipc_global_entities.next_available_ref_counters_file_->get_offset(),
      musa_ipc_global_entities.next_available_ref_counters_file_->counter_ptr(),
      device);

  musa_ipc_global_entities.next_available_ref_counters_file_->rotate_offset();
  if (!musa_ipc_global_entities.next_available_ref_counters_file_
           ->have_offsets()) {
    musa_ipc_global_entities.next_available_ref_counters_file_.reset();
  }
  return at::DataPtr(data, sent_data, MusaIPCSentDataDelete, device);
}

bool MusaIPCCollect() {
  if (!MusaIPCGlobalEntities::alive) {
    return true;
  }
  bool freed_memory = musa_ipc_global_entities.MusaIPCSentDataLimbo_.collect();
  if (musa_ipc_global_entities.MusaIPCSentDataLimbo_.size() == 0) {
    musa_ipc_global_entities.safe_clean_current_file();
  }
  return freed_memory;
}

} // namespace torch::musa

namespace c10::musa {

REGISTER_FREE_MEMORY_CALLBACK("musa_ipc_collect", MusaIPCCollectCallback);

} // namespace c10::musa
