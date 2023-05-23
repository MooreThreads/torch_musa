#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <ATen/native/ScatterGatherChecks.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

at::Tensor& GatherOut(
    const at::Tensor& input,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& out) {
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of Gather must be MUSA, but now is ",
      input.device());
  TORCH_CHECK(
      index.device().type() == kMUSA,
      "Device of index tensor of Gather must be MUSA, but now is ",
      index.device());
  TORCH_CHECK(
      out.device().type() == kMUSA,
      "Device of out tensor of Gather must be MUSA, but now is ",
      out.device());
  TORCH_CHECK(
      !sparse_grad,
      "That parameter sparse_grad of Gather is True is not supported now!");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float ||
          input.scalar_type() == at::ScalarType::Long,
      "Dtype of input tensor of Gather only support Float32 and Int64, "
      "but now it is ",
      input.scalar_type());
  c10::musa::MUSAGuard device_guard(input.device());
  auto contiguous_input = Contiguous(input);
  auto contiguous_index = Contiguous(index);

  out.resize_(contiguous_index.sizes());

  if (contiguous_index.numel() != 0) {
    TORCH_CHECK(
        contiguous_index.scalar_type() == at::ScalarType::Long,
        "gather",
        "(): Expected dtype int64 for index");
    const int64_t wrapped_dim = at::maybe_wrap_dim(dim, contiguous_input.dim());
    at::native::gather_shape_check(
        contiguous_input, wrapped_dim, contiguous_index);
  } else {
    return out;
  }

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::GatherX gather_op;
  auto mt_input = CreateMUTensor(contiguous_input);
  auto mt_index = CreateMUTensor(contiguous_index);
  auto mt_out = CreateMUTensor(out);
  CHECK_MUDNN_STATUS(gather_op.SetAxis(dim), "SetAxis");
  CHECK_MUDNN_STATUS(
      gather_op.SetMode(::musa::dnn::GatherX::Mode::GATHER_ELEMENTS),
      "SetMode");
  CHECK_MUDNN_STATUS(gather_op.Run(h, mt_out, mt_index, mt_input), "Run");
  return out;
}

at::Tensor Gather(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad) {
  Tensor result = at::empty(index.sizes(), self.options());
  GatherOut(self, dim, index, sparse_grad, result);
  return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("gather.out", &GatherOut);
  m.impl("gather", &Gather);
}

} // namespace musa
} // namespace at
