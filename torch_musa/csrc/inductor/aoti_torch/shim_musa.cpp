// clang-format off
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
// clang-format on

#include <torch_musa/csrc/core/MUSAGuard.h>
#include <torch_musa/csrc/core/MUSAStream.h>
#include "torch_musa/csrc/core/MUSACachingAllocator.h"

AOTITorchError aoti_torch_create_musa_guard(
    int32_t device_index,
    MUSAGuardHandle* ret_guard // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::musa::MUSAGuard* guard = new c10::musa::MUSAGuard(device_index);
    *ret_guard = reinterpret_cast<MUSAGuardHandle>(guard);
  });
}

AOTITorchError aoti_torch_delete_musa_guard(MUSAGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<c10::musa::MUSAGuard*>(guard); });
}

AOTITorchError aoti_torch_musa_guard_set_index(
    MUSAGuardHandle guard,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    reinterpret_cast<c10::musa::MUSAGuard*>(guard)->set_index(device_index);
  });
}

AOTITorchError aoti_torch_create_musa_stream_guard(
    void* stream,
    int32_t device_index,
    MUSAStreamGuardHandle* ret_guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::musa::MUSAStreamGuard* guard =
        new c10::musa::MUSAStreamGuard(c10::musa::getStreamFromExternal(
            static_cast<musaStream_t>(stream), device_index));
    *ret_guard = reinterpret_cast<MUSAStreamGuardHandle>(guard);
  });
}

AOTITorchError aoti_torch_delete_musa_stream_guard(
    MUSAStreamGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<c10::musa::MUSAStreamGuard*>(guard); });
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_musa_stream(int32_t device_index, void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *(musaStream_t*)(ret_stream) =
        c10::musa::getCurrentMUSAStream(device_index);
  });
}

AOTITorchError aoti_torch_musa_caching_allocator_raw_alloc(
    uint64_t nbytes,
    void** ret_ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (nbytes == 0) {
      *ret_ptr = nullptr;
      return AOTI_TORCH_SUCCESS;
    }

    *ret_ptr = c10::musa::MUSACachingAllocator::raw_alloc(nbytes);

    if (*ret_ptr == nullptr) {
      TORCH_CHECK(
          false,
          "Failed to allocate ",
          nbytes,
          " bytes from MUSA caching allocator");
    }
  });
}

AOTITorchError aoti_torch_musa_caching_allocator_raw_delete(void* ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (ptr != nullptr) {
      c10::musa::MUSACachingAllocator::raw_delete(ptr);
    }
  });
}