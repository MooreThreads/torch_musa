#ifndef TORCH_MUSA_CSRC_ATEN_UTILS_COMPLEXHELPER_H_
#define TORCH_MUSA_CSRC_ATEN_UTILS_COMPLEXHELPER_H_

#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/view_as_complex_native.h>
#include <ATen/ops/view_as_real_native.h>
#endif

#include <torch/library.h>

namespace at {
namespace musa {

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("view_as_complex", &at::native::view_as_complex);
  m.impl("view_as_real", &at::native::view_as_real);
}

} // namespace musa
} // namespace at
#endif // TORCH_MUSA_CSRC_ATEN_UTILS_COMPLEXHELPER_H_