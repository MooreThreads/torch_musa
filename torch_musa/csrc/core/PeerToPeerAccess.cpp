#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <vector>

#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/PeerToPeerAccess.h"
#include "torch_musa/csrc/utils/musa_lazy_init.h"

namespace at {
namespace musa {

static std::vector<int8_t> p2pAccessEnabled_;
static int64_t num_devices_ = -1;

namespace detail {

void init_p2p_access_cache(int64_t num_devices) {
  // p2pAccessEnabled records if p2p copies are allowed between pairs of
  // devices. Values include "1" (copy allowed), "0" (copy not allowed), and
  // "-1" (unknown).
  p2pAccessEnabled_.clear();
  p2pAccessEnabled_.resize(num_devices * num_devices, -1);
  num_devices_ = num_devices;

  for (const auto i : c10::irange(num_devices)) {
    p2pAccessEnabled_[i * num_devices + i] = 1;
  }
}

} // namespace detail

bool get_p2p_access(int dev, int dev_to_access) {
  torch::utils::musa_lazy_init();
  TORCH_CHECK(dev >= 0 || dev < num_devices_, dev, " is not a device");
  TORCH_CHECK(
      dev_to_access >= 0 || dev_to_access < num_devices_,
      dev_to_access,
      " is not a device");
  TORCH_INTERNAL_ASSERT(num_devices_ >= 0, "p2p access cache not initialized");

  // note: MUSACachingAllocator not impl needsPoolSpecificPeerAccess API, so set
  // default false.
  bool needs_pool_specific_peer_access = false;

  auto& cache = p2pAccessEnabled_[dev * num_devices_ + dev_to_access];

  if (cache != -1) {
    return cache;
  }

  c10::musa::MUSAGuard device_guard(dev);

  int access = 0;
  TORCH_MUSA_CHECK(musaDeviceCanAccessPeer(&access, dev, dev_to_access));
  if (access) {
    if (needs_pool_specific_peer_access) {
      // TODO(MT-AI): musa don't impl musaDeviceGetDefaultMemPool,
      // musaMemPoolSetAccess API. so assert this condition directly.
      TORCH_INTERNAL_ASSERT(false);
    } else {
      musaError_t err = musaDeviceEnablePeerAccess(dev_to_access, 0);
      if (err == musaErrorPeerAccessAlreadyEnabled) {
        // ignore and clear the error if access was already enabled
        musaGetLastError();
      } else {
        TORCH_MUSA_CHECK(err);
      }
    }
    cache = 1;
  } else {
    cache = 0;
  }
  return cache;
}

} // namespace musa
} // namespace at
