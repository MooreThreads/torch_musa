#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/OneHot.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace native {

DEFINE_DISPATCH(onehot_stub);
REGISTER_NO_CPU_DISPATCH(onehot_stub);

} // namespace native

namespace musa {

Tensor OneHot(const Tensor& self, int64_t num_classes) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of Bucketize must be MUSA, but now is ",
      self.device());
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Long ||
          self.scalar_type() == at::ScalarType::Int,
      "OneHot supports dtypes of float32, int32 and int64, but "
      "now it is ",
      self.scalar_type());

  c10::musa::MUSAGuard device_guard(self.device());
  auto shape = self.sizes().vec();

  // empty tensor could be converted to on hot representation,
  // but shape inference is not possible.
  if (self.numel() == 0) {
    if (num_classes <= 0) {
      AT_ERROR("Can not infer total number of classes from empty tensor.");
    } else {
      shape.push_back(num_classes);
      return at::empty(shape, self.options());
    }
  }

  if (num_classes == -1) {
    num_classes = self.max().item().toLong() + 1;
  } else {
    TORCH_CHECK(num_classes >= 1, "num_classes should be positive");
  }

  shape.push_back(num_classes);
  Tensor ret = at::zeros(
      shape, self.options().memory_format(c10::MemoryFormat::Contiguous));

  // onehot kernel dosen't support un-contiguous tensors currently
  Tensor self_contiguous = self.contiguous();

  at::native::onehot_stub(kMUSA, ret, self_contiguous, num_classes);

  return ret;
}

ADVANCED_REGISTER(aten, PrivateUse1, "one_hot", OneHot)

} // namespace musa
} // namespace at
