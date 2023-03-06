#include <c10/util/Backtrace.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <musa_runtime.h>

#include "torch_musa/csrc/core/MUSADeviceAssertionHost.h"
#include "torch_musa/csrc/core/MUSAException.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#define C10_MUSA_CHECK_WO_DSA(EXPR)                                 \
  do {                                                              \
    const musaError_t __err = EXPR;                                 \
    c10::musa::c10_musa_check_implementation(                       \
        static_cast<int32_t>(__err),                                \
        __FILE__,                                                   \
        __func__, /* Line number data type not well-defined between \
                      compilers, so we perform an explicit cast */  \
        static_cast<uint32_t>(__LINE__),                            \
        false);                                                     \
  } while (0)

namespace c10 {
namespace musa {

namespace {

/// Get the number of MUSA devices
/// We need our own implementation of this function to prevent
/// an infinite initialization loop for MUSAKernelLaunchRegistry
int dsa_get_device_count() {
  int device_count = -1;
  C10_MUSA_CHECK_WO_DSA(musaGetDeviceCount(&device_count));
  return device_count;
}

bool dsa_check_if_all_devices_support_managed_memory() {
  // For cuda references:
  // https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
  return false;
}

bool env_flag_set(const char* env_var_name) {
  const char* const env_string = std::getenv(env_var_name);
  return (env_string == nullptr) ? false : std::strcmp(env_string, "0");
}

/// Deleter for UVM/managed memory pointers
void uvm_deleter(DeviceAssertionsData* uvm_assertions_ptr) {
  // Ignore error in destructor
  if (uvm_assertions_ptr) {
    C10_MUSA_IGNORE_ERROR(musaFree(uvm_assertions_ptr));
  }
}

} // namespace

/// Check that kernels ran correctly by checking the message buffer. BLOCKING.
std::string c10_retrieve_device_side_assertion_info() {
  TORCH_WARN(
      "Funtions c10_retrieve_device_side_assertion_info() is not implemented now!");
  return "";
}

MUSAKernelLaunchRegistry::MUSAKernelLaunchRegistry()
    : do_all_devices_support_managed_memory(
          dsa_check_if_all_devices_support_managed_memory()),
      gather_launch_stacktrace(check_env_for_enable_launch_stacktracing()),
      enabled_at_runtime(check_env_for_dsa_enabled()) {
  for (C10_UNUSED const auto _ : c10::irange(dsa_get_device_count())) {
    uvm_assertions.emplace_back(nullptr, uvm_deleter);
  }

  kernel_launches.resize(max_kernel_launches);
}

bool MUSAKernelLaunchRegistry::check_env_for_enable_launch_stacktracing()
    const {
  return env_flag_set("PYTORCH_MUSA_DSA_STACKTRACING");
}

bool MUSAKernelLaunchRegistry::check_env_for_dsa_enabled() const {
  return env_flag_set("PYTORCH_USE_MUSA_DSA");
}

uint32_t MUSAKernelLaunchRegistry::insert(
    const char* launch_filename,
    const char* launch_function,
    const uint32_t launch_linenum,
    const char* kernel_name,
    const int32_t stream_id) {
  return 0;
}

std::pair<std::vector<DeviceAssertionsData>, std::vector<MUSAKernelLaunchInfo>>
MUSAKernelLaunchRegistry::snapshot() const {
  // This is likely to be the longest-lasting hold on the mutex, but
  // we only expect it to be called in cases where we're already failing
  // and speed is no longer important
  const std::lock_guard<std::mutex> lock(read_write_mutex);

  std::vector<DeviceAssertionsData> device_assertions_data;
  for (const auto& x : uvm_assertions) {
    if (x) {
      device_assertions_data.push_back(*x);
    } else {
      device_assertions_data.emplace_back();
    }
  }

  return std::make_pair(device_assertions_data, kernel_launches);
}

DeviceAssertionsData* MUSAKernelLaunchRegistry::
    get_uvm_assertions_ptr_for_current_device() {
  return nullptr;
}

MUSAKernelLaunchRegistry& MUSAKernelLaunchRegistry::get_singleton_ref() {
  static MUSAKernelLaunchRegistry launch_registry;
  return launch_registry;
}

bool MUSAKernelLaunchRegistry::has_failed() const {
  for (const auto& x : uvm_assertions) {
    if (x && x->assertion_count > 0) {
      return true;
    }
  }
  return false;
}

} // namespace musa
} // namespace c10
