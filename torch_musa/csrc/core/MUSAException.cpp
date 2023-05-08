#include <c10/util/Exception.h>
#include <musa_runtime.h>

#include <string>

#include "torch_musa/csrc/core/MUSAException.h"

namespace c10 {
namespace musa {

void c10_musa_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const int line_number,
    const bool include_device_assertions) {
  // TODO(MTAI): implement it
  TORCH_WARN(
      "Funtions c10_musa_check_implementation() is not implemented now!");
  return;
}

} // namespace musa
} // namespace c10
