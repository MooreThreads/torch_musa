#include "torch_musa/csrc/distributed/MUSAEventCache.h"
#include <map>

namespace c10d {

MUSAEventCache::MUSAEventCache() = default;

// MUSA event is used to record the start/end of one Work.
// Instead of let the MUSA event gets destroyed, we now reuse it after the Work
// has been erased from workMetaList_.
// This is to avoid the potential deadlock caused by MUSAEventDestroy.
std::shared_ptr<at::musa::MUSAEvent> MUSAEventCache::create(bool timing) {
  // Register the deleter as a callback when the WorkMCCL object is destroyed.
  // Each deleter keeps a ref count to the cache object, so that even when
  // the thread that creates the cache is gone, the cache object won't be
  // destroyed until all the events in the cache are destroyed (ref number drops
  // to zero).
  auto deleter = [cache = shared_from_this(),
                  timing](at::musa::MUSAEvent* event) {
    std::lock_guard<std::mutex> lock(cache->cacheMutex_);
    // We put the event back to the cache deque once the WorkMCCL object is
    // destroyed.
    cache->eventsArray_[timing ? 1 : 0].push_back(event);
  };
  at::musa::MUSAEvent* event = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    auto& events = eventsArray_[timing ? 1 : 0];
    // If we still have events in the cache, we reuse it. Otherwise, we create a
    // new one.
    if (!events.empty()) {
      event = events.front();
      events.pop_front();
    } else {
      event = new at::musa::MUSAEvent(
          timing ? musaEventDefault : musaEventDisableTiming);
    }
  }
  return std::shared_ptr<at::musa::MUSAEvent>(event, std::move(deleter));
}

std::shared_ptr<MUSAEventCache> MUSAEventCache::get(at::DeviceIndex device) {
  // A per-thread singleton of device-to-MUSAEventCache map.
  // Map is needed because events cannot be reused across devices.
  // Per-thread ownership is needed to support multi-threaded case (instead of
  // multi-process case).
  static thread_local std::map<at::DeviceIndex, std::shared_ptr<MUSAEventCache>>
      cacheDeviceMap;
  // Check if device has already been in the map, if not, add a new entry
  auto it = cacheDeviceMap.find(device);
  if (it == cacheDeviceMap.end()) {
    cacheDeviceMap.emplace(device, std::make_shared<MUSAEventCache>());
  }
  return cacheDeviceMap[device];
}

} // namespace c10d
