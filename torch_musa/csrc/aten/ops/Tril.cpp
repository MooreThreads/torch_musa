#include <torch/library.h>

#include <torch_musa/csrc/aten/ops/musa/musa_ops.h>

namespace at {
namespace native {
namespace musa {

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("tril", Tril);
}

} // namespace musa
} // namespace native
} // namespace at
