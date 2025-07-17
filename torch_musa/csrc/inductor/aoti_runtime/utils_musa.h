#pragma once

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/utils.h>

#include <musa.h>
#include <musa_runtime.h>

namespace torch::aot_inductor {

inline void delete_musa_guard(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_delete_musa_guard(reinterpret_cast<MUSAGuardHandle>(ptr)));
}

inline void delete_musa_stream_guard(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_musa_stream_guard(
      reinterpret_cast<MUSAStreamGuardHandle>(ptr)));
}

class AOTIMusaGuard {
 public:
  AOTIMusaGuard(int32_t device_index) : guard_(nullptr, delete_musa_guard) {
    MUSAGuardHandle ptr = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_create_musa_guard(device_index, &ptr));
    guard_.reset(ptr);
  }

  void set_index(int32_t device_index) {
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_musa_guard_set_index(guard_.get(), device_index));
  }

 private:
  std::unique_ptr<MUSAGuardOpaque, DeleterFnPtr> guard_;
};

class AOTIMusaStreamGuard {
 public:
  AOTIMusaStreamGuard(musaStream_t stream, int32_t device_index)
      : guard_(nullptr, delete_musa_stream_guard) {
    MUSAStreamGuardHandle ptr = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_create_musa_stream_guard(stream, device_index, &ptr));
    guard_.reset(ptr);
  }

 private:
  std::unique_ptr<MUSAStreamGuardOpaque, DeleterFnPtr> guard_;
};

} // namespace torch::aot_inductor
