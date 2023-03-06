#include <c10/util/Exception.h>
#include <musa_runtime.h>

#include <string>

#include "torch_musa/csrc/core/MUSADeviceAssertionHost.h"
#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAMiscFunctions.h"

namespace c10 {
namespace musa {

void c10_musa_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const int line_number,
    const bool include_device_assertions) {
  // TODO(MTAI): implement it
  const auto musa_error = static_cast<musaError_t>(err);
  const auto musa_kernel_failure = include_device_assertions
      ? c10::musa::MUSAKernelLaunchRegistry::get_singleton_ref().has_failed()
      : false;

  if (C10_LIKELY(musa_error == musaSuccess && !musa_kernel_failure)) {
    return;
  }

  auto error_unused C10_UNUSED = musaGetLastError();
  (void)error_unused;

  std::string check_message;
#ifndef STRIP_ERROR_MESSAGES
  check_message.append("MUSA error: ");
  check_message.append(musaGetErrorString(musa_error));
  check_message.append(c10::musa::get_musa_check_suffix());
  check_message.append("\n");
  if (include_device_assertions) {
    check_message.append(c10_retrieve_device_side_assertion_info());
  } else {
    check_message.append(
        "Device-side assertions were explicitly omitted for this error check; the error probably arose while initializing the DSA handlers.");
  }
#endif

  TORCH_CHECK(false, check_message);
  return;
}

} // namespace musa
} // namespace c10
