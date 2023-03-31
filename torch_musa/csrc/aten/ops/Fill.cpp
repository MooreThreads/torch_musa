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

  ::musa::dnn::Fill f;

  set_value(f, self, value);

  muHandle h;
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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("fill_.Scalar", &Fill);
  m.impl("zero_", &Zero_);
}

} // namespace musa
} // namespace native
} // namespace at
