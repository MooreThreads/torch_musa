#include <ATen/ATen.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/count_nonzero.h>
#include <ATen/ops/count_nonzero_native.h>
#include <ATen/ops/nonzero_native.h>
#include <ATen/ops/nonzero_numpy_native.h>
#endif

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at::musa {

at::Tensor& NonzeroOut(const at::Tensor& self, at::Tensor& out) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of Nonzero must be MUSA, but now is ",
      self.device());
  TORCH_CHECK(
      out.device().type() == kMUSA,
      "Device of output tensor of Nonzero must be MUSA, but now is ",
      out.device());

  if (!self.numel()) {
    out.resize_({0, self.dim()});
    return out;
  }

  c10::musa::MUSAGuard device_guard(self.device());

  if (self.scalar_type() != at::ScalarType::Float &&
      self.scalar_type() != at::ScalarType::Bool &&
      self.scalar_type() != at::ScalarType::Byte) {
    return at::native::nonzero_out_cuda(self, out);
  }

  auto contiguous_self = self.contiguous();

  TORCH_CHECK(
      contiguous_self.numel() < std::numeric_limits<int>::max(),
      "nonzero is not supported for tensors with more than INT_MAX elements, ",
      "file a support request");
  TORCH_CHECK(
      out.dtype() == at::kLong,
      "Expected object of scalar type ",
      at::kLong,
      " as out, but got ",
      out.dtype());
  out.resize_({contiguous_self.numel(), contiguous_self.dim()});

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Nonzero op;
  auto mt_input = CreateMUTensor(contiguous_self);
  auto mt_out = CreateMUTensor(out);
  CHECK_MUDNN_STATUS(op.Run(h, mt_out, mt_input, InternalMemAlloc), "Run");
  // Note that implementation of Nonzero on MUSA is different from CUDA.
  // First we malloc sufficient memory for output tensor;
  // Then MUDNN kernel will compute the actual output shape and sync.
  // Finally, we need to reset actual output shape to out tensor.
  std::vector<int64_t> out_shape;
  CHECK_MUDNN_STATUS(mt_out.GetNdInfo(out_shape), "GetNdInfo");
  std::vector<int64_t> out_shape_int64;
  for (const auto i : out_shape) {
    out_shape_int64.push_back(static_cast<int64_t>(i));
  }
  out.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape_int64);
  return out;
}

at::Tensor Nonzero(const at::Tensor& input) {
  c10::musa::MUSAGuard device_guard(input.device());
  auto result = at::empty({0}, input.options().dtype(kLong));
  NonzeroOut(input, result);
  return result;
}

Tensor CountNonzero(const Tensor& self, IntArrayRef dims) {
  return (self != 0).sum(dims);
}

} // namespace at::musa
