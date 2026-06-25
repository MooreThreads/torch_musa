#pragma once

#ifdef USE_MCCL

#include <ATen/ATen.h>
#include "torch_musa/csrc/core/MUSAStream.h"

namespace c10d {

// Check for NaNs in a tensor on a given stream. If any are found, throw a
// device-side error.
void checkForNan(const at::Tensor& tensor, at::musa::MUSAStream& stream);

} // namespace c10d

#endif // USE_MCCL
