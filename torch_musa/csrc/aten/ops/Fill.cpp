#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

void set_value(::musa::dnn::Fill& f, Tensor& self, const Scalar& value) {
  if (self.scalar_type() == at::ScalarType::Int ||
      self.scalar_type() == at::ScalarType::Long ||
      self.scalar_type() == at::ScalarType::Bool) {
    int64_t v = value.toInt();
    f.SetValue(v);
  } else if (
      self.scalar_type() == at::ScalarType::Float ||
      self.scalar_type() == at::ScalarType::Double) {
    double v = value.toDouble();
    f.SetValue(v);
  } else {
    AT_ERROR(
        "Not supported tensor type and scalar type: ",
        self.scalar_type(),
        ", ",
        value.type());
  }
}

Tensor& Fill(Tensor& self, const Scalar& value) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::fill_(self, value);
}

Tensor& Zero_(Tensor& self) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::zero_(self);
}

void MaskedFillCall(Tensor& self, const Tensor& mask, const Scalar& value) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of MaskedFill must be MTGPU, but now is ",
      self.device());
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Double ||
          self.scalar_type() == at::ScalarType::Int ||
          self.scalar_type() == at::ScalarType::Long,
      "Dtype of input tensor of masked_fill only support ",
      "Float32/Float64/Int32/Int64, but now it is ",
      self.scalar_type());
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Bool ||
          mask.scalar_type() == at::ScalarType::Byte,
      "Dtype of mask tensor of masked_fill only support ",
      "Bool, but now it is ",
      mask.scalar_type());
  c10::musa::MUSAGuard device_guard(mask.device());
  auto contiguous_self = Contiguous(self);
  auto contiguous_mask = Contiguous(mask);
  ::musa::dnn::Fill fill_op;
  set_value(fill_op, self, value);
  muHandle& handle = GetMudnnHandle();
  auto mt_self = CreateMUTensor(contiguous_self);
  auto mt_mask = CreateMUTensor(contiguous_mask);
  CHECK_MUDNN_STATUS(fill_op.Run(handle, mt_self, mt_mask), "Run");
  if (!self.is_contiguous()) {
    self.copy_(contiguous_self);
  }
}

Tensor& MaskedFill(Tensor& self, const Tensor& mask, const Scalar& value) {
  MaskedFillCall(self, mask, value);
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("fill_.Scalar", &Fill);
  m.impl("zero_", &Zero_);
  m.impl("masked_fill_.Scalar", &MaskedFill);
}

} // namespace musa
} // namespace at
