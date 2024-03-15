#include "torch_musa/csrc/core/MUSAMiscFunctions.h"

#include <stdlib.h>

namespace c10 {
namespace musa {

const char* get_musa_check_suffix() noexcept {
  static char* device_blocking_flag = getenv("MUSA_LAUNCH_BLOCKING");
  static bool blocking_enabled =
      (device_blocking_flag && atoi(device_blocking_flag));
  if (blocking_enabled) {
    return "";
  } else {
    return "\nMUSA kernel errors might be asynchronously reported at some"
           " other API call, so the stacktrace below might be incorrect."
           "\nFor debugging consider passing MUSA_LAUNCH_BLOCKING=1.";
  }
}
std::mutex* getFreeMutex() {
  static std::mutex musa_free_mutex;
  return &musa_free_mutex;
}

} // namespace musa
} // namespace c10
