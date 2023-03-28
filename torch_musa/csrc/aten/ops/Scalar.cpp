#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#pragma GCC diagnostic pop

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace native {
namespace musa {

Scalar LocalScalarDense_(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16,
      self.scalar_type(),
      "LocalScalarDense_",
      [&] {
        scalar_t value;
        muHandle h;
        musaMemcpy(
            &value,
            self.data_ptr(),
            sizeof(value),
            musaMemcpyKind::musaMemcpyDeviceToHost);
        r = Scalar(value);
      });
  return r;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_local_scalar_dense", &LocalScalarDense_);
}

} // namespace musa
} // namespace native
} // namespace at
