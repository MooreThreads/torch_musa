#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

#include <mudnn.h>

namespace at {
namespace musa {

Scalar LocalScalarDense_(const Tensor& self) {
  Scalar r;
  c10::musa::MUSAGuard device_guard(self.device());
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16,
      self.scalar_type(),
      "LocalScalarDense_",
      [&] {
        scalar_t value;
        muHandle& h = GetMudnnHandle();
        musaMemcpy(
            &value,
            self.data_ptr(),
            sizeof(value),
            musaMemcpyKind::musaMemcpyDeviceToHost);
        r = Scalar(value);
      });
  return r;
}

ADVANCED_REGISTER(aten, PrivateUse1, "_local_scalar_dense", LocalScalarDense_)

} // namespace musa
} // namespace at
