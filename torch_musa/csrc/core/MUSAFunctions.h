#ifndef TORCH_MUSA_CSRC_CORE_MUSAFUNCTIONS_H_
#define TORCH_MUSA_CSRC_CORE_MUSAFUNCTIONS_H_
#include <c10/core/impl/GPUTrace.h>
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAException.h"

namespace c10 {
namespace musa {

enum class SyncDebugMode { L_DISABLED = 0, L_WARN, L_ERROR };

void warn_or_error_on_sync();

class WarningState {
 public:
  void set_sync_debug_mode(SyncDebugMode l) {
    sync_debug_mode = l;
  }

  SyncDebugMode get_sync_debug_mode() {
    return sync_debug_mode;
  }

 private:
  SyncDebugMode sync_debug_mode = SyncDebugMode::L_DISABLED;
};

__inline__ WarningState& warning_state() {
  static WarningState warning_state_;
  return warning_state_;
}

void __inline__ memcpy_and_sync(
    void* dst,
    const void* src,
    int64_t nbytes,
    musaMemcpyKind kind,
    musaStream_t stream) {
  if (C10_UNLIKELY(
          warning_state().get_sync_debug_mode() != SyncDebugMode::L_DISABLED)) {
    warn_or_error_on_sync();
  }
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_stream_synchronization(
        reinterpret_cast<uintptr_t>(stream));
  }

  C10_MUSA_CHECK(musaMemcpyAsync(dst, src, nbytes, kind, stream));
  C10_MUSA_CHECK(musaStreamSynchronize(stream));
}

} // namespace musa
} // namespace c10
#endif // TORCH_MUSA_CSRC_CORE_MUSAFUNCTIONS_H_
