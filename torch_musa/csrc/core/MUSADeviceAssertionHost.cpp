#include "torch_musa/csrc/core/MUSADeviceAssertionHost.h"

#include <string>

#include <c10/util/Exception.h>

namespace c10 {
namespace musa {
/// Check that kernels ran correctly by checking the message buffer. BLOCKING.
std::string c10_retrieve_device_side_assertion_info() {
  TORCH_WARN(
      "Funtions c10_retrieve_device_side_assertion_info() is not implemented now!");
  return "";
}

} // namespace musa
} // namespace c10
