#ifndef TORCH_MUSA_CSRC_CORE_MUSAGRAPHSC10UTILS_H_
#define TORCH_MUSA_CSRC_CORE_MUSAGRAPHSC10UTILS_H_

#include <utility>

// MUSA Graphs utils used by c10 and aten.
// aten/musa/MUSAGraphsUtils.cuh adds utils used by aten only.

namespace c10 {
namespace musa {

using CaptureId_t = unsigned long long;

// first is set if the instance is created by MUSAGraph::capture_begin.
// second is set if the instance is created by at::musa::graph_pool_handle.
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

enum class CaptureStatus : int { None = 0 };

inline std::ostream& operator<<(std::ostream& os, CaptureStatus status) {
  switch (status) {
    case CaptureStatus::None:
      os << "musaStreamCaptureStatusNone";
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Unknown MUSA graph CaptureStatus", int(status));
  }
  return os;
}

inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  return CaptureStatus::None;
}

} // namespace musa
} // namespace c10
#endif // TORCH_MUSA_CSRC_CORE_MUSAGRAPHSC10UTILS_H_
