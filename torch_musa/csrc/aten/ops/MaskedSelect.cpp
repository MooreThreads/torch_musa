#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace native {
namespace musa {

at::Tensor& MaskedSelectOut(
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Tensor& out) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of MaskedSelect must be MUSA, but now is ",
      self.device());
  TORCH_CHECK(
      mask.device().type() == kMUSA,
      "Device of mask tensor of MaskedSelect must be MUSA, but now is ",
      mask.device());
  TORCH_CHECK(
      out.device().type() == kMUSA,
      "Device of output tensor of MaskedSelect must be MUSA, but now is ",
      out.device());
  TORCH_CHECK(
      mask.scalar_type() == ScalarType::Byte ||
          mask.scalar_type() == ScalarType::Bool,
      "masked_select: expected BoolTensor or ByteTensor for mask,",
      " but now is ",
      mask.scalar_type());
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Int ||
          self.scalar_type() == at::ScalarType::Long,
      "Dtype of input tensor of masked_select only support ",
      "Float32/Int32/Int64, but now it is ",
      self.scalar_type());
  TORCH_CHECK(
      self.scalar_type() == out.scalar_type(),
      "masked_select(): input and result must have the same scalar type, ",
      "but now the former is : ",
      self.scalar_type(),
      ", and the latter is : ",
      out.scalar_type());
  auto mask_temp = (mask.dim() == 0)
      ? c10::MaybeOwned<Tensor>::owned(mask.unsqueeze(0))
      : c10::MaybeOwned<Tensor>::borrowed(mask);
  auto self_temp = (self.dim() == 0)
      ? c10::MaybeOwned<Tensor>::owned(self.unsqueeze(0))
      : c10::MaybeOwned<Tensor>::borrowed(self);
  if (!mask_temp->numel() || !self_temp->numel()) {
    out = at::empty({0}, self.options());
    return out;
  }

  c10::musa::MUSAGuard device_guard(mask.device());

  auto contiguous_self = Contiguous(*self_temp);
  auto contiguous_mask = Contiguous(*mask_temp);

  c10::MaybeOwned<Tensor> expand_mask, expand_input;
  std::tie(expand_mask, expand_input) =
      expand_outplace(contiguous_mask, contiguous_self);
  out.resize_({(*expand_input).numel()});
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::MaskedSelect maskedselect_op;
  auto mt_input = CreateMUTensor(*expand_input, true);
  auto mt_mask = CreateMUTensor(*expand_mask, true);
  auto mt_result = CreateMUTensor(out, true);
  CHECK_MUDNN_STATUS(
      maskedselect_op.Run(h, mt_result, mt_input, mt_mask, InternalMemAlloc),
      "Run");
  // Note that implementation of MaskedSelect on MUSA is different from CUDA.
  // First we malloc sufficient memory for output tensor;
  // Then MUDNN kernel will compute the actual output shape and sync.
  // Finally, wo need to reset actual output shape to out tensor.
  std::vector<int32_t> out_shape;
  std::vector<int32_t> out_stride;
  CHECK_MUDNN_STATUS(mt_result.GetNdInfo(out_shape, out_stride), "GetNdInfo");
  std::vector<int64_t> out_shape_int64;
  for (const auto i : out_shape) {
    out_shape_int64.push_back(static_cast<int64_t>(i));
  }
  std::vector<int64_t> out_stride_int64;
  for (const auto i : out_stride) {
    out_stride_int64.push_back(static_cast<int64_t>(i));
  }
  out.unsafeGetTensorImpl()->set_sizes_and_strides(
      out_shape_int64, out_stride_int64);
  return out;
}

at::Tensor MaskedSelect(const at::Tensor& input, const at::Tensor& mask) {
  auto result = at::empty({0}, input.options());
  MaskedSelectOut(input, mask, result);
  return result;
}

at::Tensor& NonzeroOut(const at::Tensor& self, at::Tensor& out) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of Nonzero must be MUSA, but now is ",
      self.device());
  TORCH_CHECK(
      out.device().type() == kMUSA,
      "Device of output tensor of Nonzero must be MUSA, but now is ",
      out.device());
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Bool,
      "Dtype of input tensor of NoneZero only support "
      "Float32/Bool, ",
      "but now it is ",
      self.scalar_type());
  if (!self.numel()) {
    out.resize_({0, self.dim()});
    return out;
  }

  c10::musa::MUSAGuard device_guard(self.device());
  auto contiguous_self = Contiguous(self);

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
  std::vector<int32_t> out_shape;
  CHECK_MUDNN_STATUS(mt_out.GetNdInfo(out_shape), "GetNdInfo");
  std::vector<int64_t> out_shape_int64;
  for (const auto i : out_shape) {
    out_shape_int64.push_back(static_cast<int64_t>(i));
  }
  out.unsafeGetTensorImpl()->set_sizes_contiguous(out_shape_int64);
  return out;
}

at::Tensor Nonzero(const at::Tensor& input) {
  auto result = at::empty({0}, input.options().dtype(kLong));
  NonzeroOut(input, result);
  return result;
}

at::Tensor& MaskedScatter(
    at::Tensor& self,
    const at::Tensor& mask,
    const at::Tensor& source) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of self tensor of MaskedScatter must be MUSA, but now is ",
      self.device());
  TORCH_CHECK(
      mask.device().type() == kMUSA,
      "Device of mask tensor of MaskedScatter must be MUSA, but now is ",
      mask.device());
  TORCH_CHECK(
      source.device().type() == kMUSA,
      "Device of source tensor of MaskedScatter must be MUSA, but now is ",
      source.device());
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Int ||
          self.scalar_type() == at::ScalarType::Long,
      "Dtype of self tensor of MaskedScatter only support Float32/Int32/Int64,",
      " but now it is ",
      self.scalar_type());
  at::assert_no_internal_overlap(self);
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "masked_scatter: expected self and source to have same dtypes but got ",
      self.scalar_type(),
      " and ",
      source.scalar_type());
  c10::musa::MUSAGuard device_guard(mask.device());
  auto contiguous_self = Contiguous(self);
  auto contiguous_mask = Contiguous(mask);
  c10::MaybeOwned<Tensor> b_mask =
      expand_inplace(contiguous_self, contiguous_mask, "masked_scatter_");
  if (b_mask->dtype() == ScalarType::Byte) {
    TORCH_WARN(
        "masked_scatter_ received a mask with dtype torch.uint8, this "
        "behavior is now deprecated, please use a mask with dtype torch.bool "
        "instead.");
  }
  if (self.numel() == 0) {
    return self;
  }

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::MaskedScatter op;
  auto mt_input = CreateMUTensor(contiguous_self);
  auto mt_mask = CreateMUTensor(contiguous_mask);
  auto mt_source = CreateMUTensor(source);
  CHECK_MUDNN_STATUS(
      op.Run(h, mt_input, mt_mask, mt_source, InternalMemAlloc), "Run");
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("masked_select", &MaskedSelect);
  m.impl("masked_select.out", &MaskedSelectOut);
  m.impl("nonzero", &Nonzero);
  m.impl("nonzero.out", &NonzeroOut);
  m.impl("masked_scatter_", &MaskedScatter);
}

} // namespace musa
} // namespace native
} // namespace at
