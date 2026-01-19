#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nested_from_padded_native.h>
#endif

#include "torch_musa/csrc/aten/utils/StateGuard.h"

namespace at::musa {

Tensor _NestedFromPadded(
    const Tensor& padded,
    const Tensor& sizes,
    const bool do_transform_0213) {
  MAKE_STATE_GUARD(STRICT_MASK_SELECT, true);
  return at::native::nested_from_padded_generic(
      padded, sizes, do_transform_0213);
}

} // namespace at::musa
