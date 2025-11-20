#ifndef _TORCH_MUSA_CSRC_ATEN_UTILS_CONTEXT_H_
#define _TORCH_MUSA_CSRC_ATEN_UTILS_CONTEXT_H_

#include <c10/core/ScalarType.h>
#include <c10/util/CallOnce.h>
#include <torch/csrc/tensor/python_tensor.h>

#include <mudnn.h>

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAHooksInterface.h"

namespace at {
namespace musa {

// Create a `Context` class to track the global information specified/queried by
// users.
class Context {
 public:
  Context() = default;

  static bool HasMUSA() {
    return at::detail::getMUSAHooks().hasMUSA();
  }

  void LazyInitMUSA() {
    c10::call_once(musa_init_, [&] { at::detail::getMUSAHooks().initMUSA(); });
  }

  // Helper function to check if TF32 is enabled or not.
  bool GetAllowTF32() const;

  // Set the allow_tf32 flag.
  void SetAllowTF32(bool allow_tf32);

 private:
  // TF32 is enabled by default to keep consistent to official PyTorch.
  bool allow_tf32_ = true;
  c10::once_flag musa_init_;
};

// A global singleton for `Context`.
Context& GlobalContext();

static inline void Init() {
  GlobalContext();
}

static inline bool HasMUSA() {
  return GlobalContext().HasMUSA();
}

// Get the ComputeMode (TENSOR/SCALAR) from the context and input tensor dtype
::musa::dnn::Convolution::ComputeMode GetComputeModeFromCtx(
    const at::ScalarType& dtype);

PyMethodDef* GetContextMethods();

} // namespace musa
} // namespace at

#endif //_TORCH_MUSA_CSRC_ATEN_UTILS_CONTEXT_H_
