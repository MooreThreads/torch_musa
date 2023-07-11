// Functions that fill Tensors with constants.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Fill.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/quantized/QTensor.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <torch/library.h>
namespace at {
namespace native {

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

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("fill_.Scalar", TORCH_FN(FillQuantizedScalar));
  m.impl("fill_.Tensor", TORCH_FN(FillQuantizedTensor));
  m.impl("masked_fill_.Scalar", TORCH_FN(MaskedFillQuantize));
  m.impl("masked_fill_.Tensor", TORCH_FN(MaskedFillQuantizeTensor));
}

} // namespace native
} // namespace at
