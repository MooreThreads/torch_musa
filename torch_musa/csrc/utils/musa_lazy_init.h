#ifndef TORCH_MUSA_CSRC_UTILS_MUSA_LAZY_INIT_H_
#define TORCH_MUSA_CSRC_UTILS_MUSA_LAZY_INIT_H_
#include <c10/core/TensorOptions.h>

namespace torch {
namespace utils {

// The INVARIANT is that this function MUST be called before you attempt
// to get a MUSA Type object from torch_musa, in any way.
//
void musa_lazy_init();
void set_requires_musa_init(bool value);

static void maybe_initialize_musa(const at::TensorOptions& options) {
  if (options.device().is_privateuseone()) {
    torch::utils::musa_lazy_init();
  }
}

} // namespace utils
} // namespace torch

#endif // TORCH_MUSA_CSRC_UTILS_MUSA_LAZY_INIT_H_
