// Functions that fill Tensors with constants.
#include <ATen/Config.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Fill.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/masked_fill_native.h>
#include <ATen/ops/ones.h>
#endif

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/quantized/QTensor.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
static Tensor& FillOutQuantized(Tensor& self, const Scalar& value) {
  at::Tensor out = at::ones(self.sizes()).to(kFloat) * value;
  out = out.to(self.device()).to(self.suggest_memory_format());
  // TODO(@songlin): Fix copy_ error of musa quantized tensor
  self = QTensorCopy(self, out);
  return self;
}

Tensor& FillQuantizedScalar(Tensor& self, const Scalar& value) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return FillOutQuantized(self, value);
}

Tensor& FillQuantizedTensor(Tensor& self, const Tensor& value) {
  const OptionalDeviceGuard device_guard(device_of(self));
  TORCH_CHECK(
      value.dim() == 0,
      "fill_ only supports 0-dimension value tensor but got tensor with ",
      value.dim(),
      " dimensions.");
  return FillOutQuantized(self, value.item());
}

Tensor& MaskedFillQuantize(
    Tensor& self,
    const Tensor& mask,
    const Scalar& value) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::masked_fill__quantized_cuda(self, mask, value);
}

Tensor& MaskedFillQuantizeTensor(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::masked_fill__quantized_cuda(self, mask, value);
}

} // namespace musa
} // namespace at
