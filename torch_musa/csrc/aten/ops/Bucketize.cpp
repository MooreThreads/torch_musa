#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/Bucketize.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace native {

DEFINE_DISPATCH(bucketize_stub);
REGISTER_NO_CPU_DISPATCH(bucketize_stub);

} // namespace native

namespace musa {

Tensor Bucketize(
    const Tensor& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of Bucketize must be MUSA, but now is ",
      self.device());
  TORCH_CHECK(
      boundaries.device().type() == kMUSA,
      "Device of boundaries tensor of Bucketize must be MUSA, but now is ",
      boundaries.device());
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Long ||
          self.scalar_type() == at::ScalarType::Int,
      "Bucketize supports dtypes of float32, int32 and int64, but "
      "now it is ",
      self.scalar_type());

  c10::musa::MUSAGuard device_guard(self.device());

  // bucketize kernel dosen't support un-contiguous tensors currently
  Tensor self_contiguous = self.contiguous();
  Tensor boundaries_contiguous = boundaries.contiguous();

  at::ScalarType out_dtype =
      out_int32 ? at::ScalarType::Int : at::ScalarType::Long;
  Tensor out = at::empty_like(
      self,
      self.options().dtype(out_dtype).memory_format(
          at::MemoryFormat::Contiguous));

  at::native::bucketize_stub(
      kMUSA, out, self_contiguous, boundaries_contiguous, right);

  return out;
}

ADVANCED_REGISTER(aten, PrivateUse1, "bucketize.Tensor", Bucketize)

} // namespace musa
} // namespace at
