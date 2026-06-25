#include <ATen/ATen.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/count_nonzero.h>
#include <ATen/ops/count_nonzero_native.h>
#endif

#include <ATen/native/TensorConversions.h>
#include "torch_musa/csrc/aten/utils/StateGuard.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at::musa {

at::Tensor MulSparseCsr(const at::Tensor& self, const at::Tensor& other) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  MAKE_STATE_GUARD(NNZ_NONE_CONTIGUOUS, true);
  return at::native::mul_sparse_csr(self, other);
}

at::Tensor& MulOutSparseCsr(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  MAKE_STATE_GUARD(NNZ_NONE_CONTIGUOUS, true);
  return at::native::mul_out_sparse_csr(self, other, out);
}

at::Tensor& MulSparseCsr_(at::Tensor& self, const at::Tensor& other) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  MAKE_STATE_GUARD(NNZ_NONE_CONTIGUOUS, true);
  return at::native::mul_sparse_csr_(self, other);
}
} // namespace at::musa
