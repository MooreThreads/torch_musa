#ifndef TORCH_MUSA_CSRC_CORE_POOL_H_
#define TORCH_MUSA_CSRC_CORE_POOL_H_

#include "torch_musa/csrc/core/MUSAEvent.h"
#include "torch_musa/csrc/core/MUSAFunctions.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <musa_runtime_api.h>

#include <memory>
#include <mutex>
#include <vector>

namespace at::musa::detail {

template <typename E, bool Lazy>
class EventPoolBase;

template <>
class EventPoolBase<musaEvent_t, false> {
 protected:
  auto CreateWithFlags(unsigned int flags) {
    auto ptr = std::make_unique<musaEvent_t>();
    C10_MUSA_CHECK(musaEventCreateWithFlags(ptr.get(), flags));
    return ptr;
  }
};

template <bool Lazy>
class EventPoolBase<MUSAEvent, Lazy> {
 protected:
  auto CreateWithFlags(unsigned int flags) {
    auto ptr = std::make_unique<MUSAEvent>(flags);
    if constexpr (!Lazy) {
      ptr->createEvent(getCurrentMUSAStream().device_index());
    }
    return ptr;
  }
};

template <typename E, bool Lazy>
class EventPool : public EventPoolBase<E, Lazy> {
 public:
  using Event = std::unique_ptr<E, std::function<void(E*)>>;
  EventPool() : pools_(device_count()) {}

  Event get(DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<DeviceIndex>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](E* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<E>(event));
    };

    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }

    auto new_ptr = this->CreateWithFlags(musaEventDisableTiming);
    return Event(new_ptr.release(), destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<E>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

} // namespace at::musa::detail

#endif // TORCH_MUSA_CSRC_CORE_POOL_H_
