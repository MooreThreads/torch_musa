#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

#include <mudnn.h>

namespace at {
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
      self.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of Bucketize only support Float32, but "
      "now it is ",
      self.scalar_type());

  c10::musa::MUSAGuard device_guard(self.device());
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Bucketize mBucketize;
  CHECK_MUDNN_STATUS(mBucketize.SetRight(right), "SetRight");

  auto self_contiguous = self.contiguous();
  auto self_input = CreateMUTensor(self_contiguous);

  auto boundaries_contiguous = boundaries.contiguous();
  auto boundaries_input = CreateMUTensor(boundaries_contiguous);
  // note: Unsupported out data type: FLOAT in muDNN, hence out_int32 is dummy
  Tensor out = at::empty_like(
      self,
      self.options().dtype(at::kLong).memory_format(
          at::MemoryFormat::Contiguous));

  auto output = CreateMUTensor(out);
  CHECK_MUDNN_STATUS(
      mBucketize.Run(h, output, self_input, boundaries_input), "Run");
  return out_int32 ? out.to(at::kInt) : out;
}

ADVANCED_REGISTER(aten, PrivateUse1, "bucketize.Tensor", Bucketize)

} // namespace musa
} // namespace at
