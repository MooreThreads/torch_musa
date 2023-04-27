#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace native {
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

// TODO(yao-wang): Fill is not in the list of long-term-supporting ops of
// muDNN. We port this op with MUSA toolkit.
void FillCall(Tensor& self, const Scalar& value) {
  if (!self.is_contiguous()) {
    AT_ERROR("Fill op doesn't support non-contiguous tensor now");
  }

  torch_musa::MUSAGuard device_guard(self.device());
  ::musa::dnn::Fill f;

  set_value(f, self, value);

  muHandle& h = getMudnnHandle();
  // When self tensor is contiguous but storage_offset is not 0,
  // first creates a new contiguous tensor, then fill value in the new tensor,
  // copy new tensor value back to self tensor at last.
  if (self.storage_offset()) {
    auto new_contiguous_tensor = at::empty(
        self.sizes(), self.options().memory_format(MemoryFormat::Contiguous));
    auto out = CreateMUTensor(new_contiguous_tensor);
    CHECK_MUDNN_STATUS(f.Run(h, out), "Run");
    self.copy_(new_contiguous_tensor);
  } else {
    auto out = CreateMUTensor(self);
    CHECK_MUDNN_STATUS(f.Run(h, out), "Run");
  }
}

Tensor& Fill(Tensor& self, const Scalar& value) {
  FillCall(self, value);
  return self;
}

Tensor& Zero_(Tensor& self) {
  FillCall(self, 0);
  return self;
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
  torch_musa::MUSAGuard device_guard(mask.device());
  auto contiguous_self = Contiguous(self);
  auto contiguous_mask = Contiguous(mask);
  ::musa::dnn::Fill fill_op;
  set_value(fill_op, self, value);
  muHandle& handle = getMudnnHandle();
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
} // namespace native
} // namespace at
