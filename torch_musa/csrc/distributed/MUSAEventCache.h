#pragma once

#include <array>
#include <deque>
#include <memory>
#include <mutex>

#include "torch_musa/csrc/core/MUSAEvent.h"

namespace c10d {

class MUSAEventCache : public std::enable_shared_from_this<MUSAEventCache> {
 public:
  MUSAEventCache();
  std::shared_ptr<at::musa::MUSAEvent> create(bool timing);
  static std::shared_ptr<MUSAEventCache> get(at::DeviceIndex device);

 private:
  std::mutex cacheMutex_;
  // NOTE: We intentionally store raw pointers so that
  // we do not attempt to destroy the event objects on process exit,
  // because musa may be gone.
  std::array<std::deque<at::musa::MUSAEvent*>, 2>
      eventsArray_; // 0 for timing=false, 1 for timing=true
};

} // namespace c10d
