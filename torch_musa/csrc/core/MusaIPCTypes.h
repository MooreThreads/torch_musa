#ifndef TORCH_MUSA_CSRC_CORE_MUSAIPCTYPES_H_
#define TORCH_MUSA_CSRC_CORE_MUSAIPCTYPES_H_
#include <c10/core/Allocator.h>
#include <c10/util/Logging.h>
#include <torch/csrc/Export.h>
#include <cstddef>

#include "musa_runtime_api.h"
#include "torch_musa/csrc/core/Allocator.h"
namespace torch {
namespace musa {
bool MusaIPCCollect();

struct MusaIPCReceivedData final {
  MusaIPCReceivedData() = default;
  explicit MusaIPCReceivedData(std::shared_ptr<void> shared_ptr)
      : shared_ptr_(std::move(shared_ptr)) {}
  std::shared_ptr<void> shared_ptr_;
};

struct MusaIPCSentData final {
  std::string handle_;
  uint64_t offset_;
  uint64_t* counter_ptr_; // Reference counter shared memory block
  at::DataPtr original_ptr_; // Original mem allocation
  musaEvent_t event_; // Sync cuEventDestroy
  bool event_sync_required_;
  at::Device device_;

  MusaIPCSentData(
      std::string handle,
      uint64_t offset,
      uint64_t* counter_ptr,
      at::Device device);
  ~MusaIPCSentData();

  uint64_t counter_value();
  std::string handle() {
    return handle_;
  }
  uint64_t offset() {
    return offset_;
  }
  void set_original_ptr(at::DataPtr data_ptr) {
    original_ptr_ = std::move(data_ptr);
  }
};

at::DataPtr GetNewRefCountedSentData(void* data, at::Device device);

inline constexpr int64_t MUSA_IPC_REF_COUNTER_FILE_SIZE = 10000;
inline constexpr int64_t MUSA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO = 1000;
// This was determined empirically that MUSA have the limit
// on the number of recorded blocking interprocess events. It is around ~22,000.
// And to give us leeway, we picked 1000 as it gives us enough events to share
// tensors effectively.
inline constexpr int64_t MUSA_IPC_MAXIMUM_EVENTS_TO_USE = 1000;

// All to be deleted data blocks with non zero reference counter goes there
struct MusaIPCSentDataLimbo final {
  ~MusaIPCSentDataLimbo();
  bool collect();
  void add(std::unique_ptr<MusaIPCSentData> shared_block);
  uint64_t size();

 private:
  std::vector<std::unique_ptr<MusaIPCSentData>> shared_blocks_;
  std::mutex limbo_mutex_;
};

struct MusaIPCRefCountersFile final {
  MusaIPCRefCountersFile(
      std::string handle,
      uint64_t size,
      at::DataPtr data_ptr)
      : next_offset_(0),
        size_(size),
        used_slots_(0),
        handle_(std::move(handle)),
        refcounted_shared_mem_(std::move(data_ptr)) {}

  uint64_t* counter_ptr() {
    return static_cast<uint64_t*>(refcounted_shared_mem_.get()) + next_offset_;
  }

  void set_counter(uint64_t value) {
    *counter_ptr() = value;
  }

  bool have_offsets() {
    return next_offset_ < size_;
  }

  bool offsets_in_use() {
    return used_slots_;
  }

  uint64_t get_offset() {
    return next_offset_;
  }

  void rotate_offset() {
    next_offset_++;
    used_slots_++;
  }

  void return_offset(uint64_t offset /* unused */) {
    used_slots_--;
  }

  std::string handle() {
    return handle_;
  }

 private:
  uint64_t next_offset_;
  uint64_t size_;
  uint64_t used_slots_;
  std::string handle_;
  at::DataPtr refcounted_shared_mem_;
};

} // namespace musa
} // namespace torch

namespace c10 {
namespace musa {
class MusaIPCCollectCallback : public FreeMemoryCallback {
 public:
  bool Execute() override {
    return torch::musa::MusaIPCCollect();
  }
};

} // namespace musa
} // namespace c10
#endif // TORCH_MUSA_CSRC_CORE_MUSAIPCTYPES_H_
