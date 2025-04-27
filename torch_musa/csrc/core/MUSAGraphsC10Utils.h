#ifndef TORCH_MUSA_CSRC_CORE_MUSAGRAPHSC10UTILS_H_
#define TORCH_MUSA_CSRC_CORE_MUSAGRAPHSC10UTILS_H_

#include <c10/musa/MUSA_PORT_Macros.h>
#include "torch_musa/csrc/core/MUSAStream.h"

#include <musa_runtime_api.h>

namespace c10 {
namespace musa {

using CaptureId_t = unsigned long long;

// first is set if the instance is created by CUDAGraph::capture_begin.
// second is set if the instance is created by at::musa::graph_pool_handle.
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

struct C10_MUSA_API MUSAStreamCaptureModeGuard {
  MUSAStreamCaptureModeGuard(musaStreamCaptureMode desired) {
    strictness_ = desired;
    C10_MUSA_CHECK(musaThreadExchangeStreamCaptureMode(&strictness_));
  }
  ~MUSAStreamCaptureModeGuard() {
    C10_MUSA_CHECK(musaThreadExchangeStreamCaptureMode(&strictness_));
  }

 private:
  musaStreamCaptureMode strictness_;
};

enum class CaptureStatus : int {
  None = int(musaStreamCaptureStatus::musaStreamCaptureStatusNone),
  Active = int(musaStreamCaptureStatus::musaStreamCaptureStatusActive),
  Invalidated = int(musaStreamCaptureStatus::musaStreamCaptureStatusInvalidated)
};

inline std::ostream& operator<<(std::ostream& os, CaptureStatus status) {
  switch (status) {
    case CaptureStatus::None:
      os << "musaStreamCaptureStatusNone";
      break;
    case CaptureStatus::Active:
      os << "musaStreamCaptureStatusActive";
      break;
    case CaptureStatus::Invalidated:
      os << "musaStreamCaptureStatusInvalidated";
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Unknown MUSA graph CaptureStatus", int(status));
  }
  return os;
}

// Use this version where you're sure a MUSA context exists already.
inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  musaStreamCaptureStatus is_capturing;
#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION > 4001)
  C10_MUSA_CHECK(
      musaStreamIsCapturing(c10::musa::getCurrentMUSAStream(), &is_capturing));
  return CaptureStatus(is_capturing);
#endif
  return CaptureStatus::None;
}

} // namespace musa
} // namespace c10
#endif // TORCH_MUSA_CSRC_CORE_MUSAGRAPHSC10UTILS_H_
