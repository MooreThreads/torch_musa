#include "torch_musa/csrc/core/MUSAFunctions.h"

#include <c10/util/Exception.h>

namespace c10 {
namespace musa {
// this function has to be called from callers performing musa synchronizing
// operations, to raise proper error or warning
void warn_or_error_on_sync() {
  if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
    TORCH_CHECK(false, "called a synchronizing MUSA operation");
  } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
    TORCH_WARN("called a synchronizing MUSA operation");
  }
}

} // namespace musa
} // namespace c10
