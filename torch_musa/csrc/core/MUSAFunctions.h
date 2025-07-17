#ifndef TORCH_MUSA_CSRC_CORE_MUSAFUNCTIONS_H_
#define TORCH_MUSA_CSRC_CORE_MUSAFUNCTIONS_H_

#include <c10/core/impl/GPUTrace.h>

#include <musa_runtime_api.h>

#include "torch_musa/csrc/core/MUSAException.h"

namespace c10::musa {

// returns -1 on failure
int32_t GetDriverVersion();

bool is_musa_available();

DeviceIndex device_count() noexcept;

DeviceIndex device_count_ensure_non_zero();

musaError_t GetDeviceCount(int* dev_count);

musaError_t GetDevice(DeviceIndex* device);

musaError_t SetDevice(DeviceIndex device);

musaError_t MaybeSetDevice(DeviceIndex device);

DeviceIndex ExchangeDevice(DeviceIndex to_device);

DeviceIndex MaybeExchangeDevice(DeviceIndex to_device);

void SetTargetDevice();

bool hasPrimaryContext(DeviceIndex device_index);

void device_synchronize();

void warn_or_error_on_sync();

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

inline WarningState& warning_state() {
  static WarningState warning_state_;
  return warning_state_;
}

inline void memcpy_and_sync(
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
        c10::kPrivateUse1, reinterpret_cast<uintptr_t>(stream));
  }

  C10_MUSA_CHECK(musaMemcpyAsync(dst, src, nbytes, kind, stream));
  C10_MUSA_CHECK(musaStreamSynchronize(stream));
}

inline void stream_synchronize(musaStream_t stream) {
  if (C10_UNLIKELY(
          warning_state().get_sync_debug_mode() != SyncDebugMode::L_DISABLED)) {
    warn_or_error_on_sync();
  }
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_stream_synchronization(
        c10::kPrivateUse1, reinterpret_cast<uintptr_t>(stream));
  }
  C10_MUSA_CHECK(musaStreamSynchronize(stream));
}

Device getDeviceFromPtr(void* ptr);

std::optional<DeviceIndex> getDeviceIndexWithPrimaryContext();

bool isPinnedPtr(const void* data);

} // namespace c10::musa

#endif // TORCH_MUSA_CSRC_CORE_MUSAFUNCTIONS_H_
