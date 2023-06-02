#ifndef TORCH_MUSA_CSRC_CORE_MUSAFUNCTIONS_H_
#define TORCH_MUSA_CSRC_CORE_MUSAFUNCTIONS_H_
#include "torch_musa/csrc/core/Device.h"

namespace c10 {
namespace musa {

enum class SyncDebugMode { L_DISABLED = 0, L_WARN, L_ERROR };

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

} // namespace musa
} // namespace c10
#endif // TORCH_MUSA_CSRC_CORE_MUSAFUNCTIONS_H_
