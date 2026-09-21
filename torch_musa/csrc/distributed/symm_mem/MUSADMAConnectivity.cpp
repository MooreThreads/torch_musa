#include <torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp>

#include <c10/util/Logging.h>
#include <utility>
#include <vector>

#include "torch_musa/csrc/core/MUSAFunctions.h"

namespace {

constexpr const char* kMtLinkConnectionType = "mtlink";

struct C10_EXPORT MUSAMtLinkDetector : public c10d::DMAConnectivityDetector {
  c10::intrusive_ptr<c10d::DMAConnectivity> detect() override {
    const auto numDevices = static_cast<int>(c10::musa::device_count());
    std::vector<std::vector<int>> matrix(
        numDevices, std::vector<int>(numDevices, 0));

    for (int i = 0; i < numDevices; ++i) {
      for (int j = 0; j < numDevices; ++j) {
        if (i == j) {
          continue;
        }
        int mtlinkPortCount = 0;
        auto err = C10_MUSA_ERROR_HANDLED(musaDeviceGetP2PAttribute(
            &mtlinkPortCount, musaDevP2PAttrMtlinkPortCount, i, j));
        if (err == musaSuccess) {
          matrix[i][j] = mtlinkPortCount;
        } else {
          (void)musaGetLastError();
          LOG(WARNING)
              << "MUSAMtLinkDetector: failed to query MTLink port count "
              << "from device " << i << " to device " << j
              << ". Assuming no MTLink connection. " << musaGetErrorString(err);
        }
      }
    }

    return c10::make_intrusive<c10d::DMAConnectivity>(
        c10::DeviceType::PrivateUse1, kMtLinkConnectionType, std::move(matrix));
  }
};

struct RegisterDetector {
  RegisterDetector() {
    c10d::register_dma_connectivity_detector(
        c10::DeviceType::PrivateUse1,
        kMtLinkConnectionType,
        c10::make_intrusive<MUSAMtLinkDetector>());
  }
};

static RegisterDetector register_detector_;

} // namespace
