#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/BinaryOps.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/full_like.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/result_type_native.h>
#include <ATen/ops/where_native.h>
#endif

#include <ATen/TensorIterator.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {
using TERNARY_MODE = ::musa::dnn::Ternary::Mode;

void TernaryCall(
    Tensor& output,
    const Tensor& self,
    const Tensor& input1,
    const Tensor& input2,
    TERNARY_MODE m,
    const Scalar& alpha_scalar) {
  if (C10_UNLIKELY(
          input1.numel() == 0 || input2.numel() == 0 || self.numel() == 0)) {
    return;
  }
  c10::musa::MUSAGuard device_guard(self.device());
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Ternary top;

  if (!alpha_scalar.equal(1)) {
    if (m == TERNARY_MODE::ADDCMUL) {
      m = TERNARY_MODE::ADDCMUL_ALPHA;
    } else if (m == TERNARY_MODE::ADDCDIV) {
      m = TERNARY_MODE::ADDCDIV_ALPHA;
    }
    if (self.is_floating_point()) {
      CHECK_MUDNN_STATUS(top.SetAlpha(alpha_scalar.toDouble()), "SetAlpha");
    } else {
      CHECK_MUDNN_STATUS(top.SetAlpha(alpha_scalar.toLong()), "SetAlpha");
    }
  }

  auto input1_mt = CreateMUTensor(input1);
  auto input2_mt = CreateMUTensor(input2);
  auto self_mt = CreateMUTensor(self);
  // output should be contiguous, caller should be responsible for this
  auto om_mt = CreateMUTensor(output);
  CHECK_MUDNN_STATUS(top.SetMode(m), "SetMode");
  CHECK_MUDNN_STATUS(top.Run(h, om_mt, self_mt, input1_mt, input2_mt), "Run");
}

/**
 * @brief Ternary operations convention. All of the function params that are
 * tensors could be non-contiguous, be careful.
 *
 * @param self input tensor
 * @param input1 first of the two operands
 * @param input2 second of the two operands
 * @param alpha_scalar scaling factor
 * @param output output tensor to write result
 * @param m Ternary Mode Type.
 */
void TernarycommonDtypeCall(
    const Tensor& self,
    const Tensor& input1,
    const Tensor& input2,
    Scalar const& alpha_scalar,
    Tensor& output,
    TERNARY_MODE m) {
  auto common_dtype = at::result_type(input1, input2);
  at::native::alpha_check(common_dtype, alpha_scalar);

  // WARN: output created by torch, which could be non-contiguous.
  Tensor contiguous_output =
      FormatContiguous(output, at::MemoryFormat::Contiguous);

  Tensor contiguous_self = FormatContiguous(self, at::MemoryFormat::Contiguous);
  Tensor contiguous_input1 =
      FormatContiguous(input1, at::MemoryFormat::Contiguous);
  Tensor contiguous_input2 =
      FormatContiguous(input2, at::MemoryFormat::Contiguous);

  TernaryCall(
      contiguous_output,
      contiguous_self,
      contiguous_input1,
      contiguous_input2,
      m,
      alpha_scalar);

  if (output.data_ptr() != contiguous_output.data_ptr()) {
    output.copy_(contiguous_output);
  }
}

Tensor& TernaryOut(
    Tensor& output,
    const Tensor& cond,
    const Tensor& self,
    const Tensor& other,
    TERNARY_MODE m,
    const Scalar& alpha_scalar) {
  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "input scalar type must the same");

  Tensor contiguous_out =
      FormatContiguous(output, at::MemoryFormat::Contiguous);
  Tensor contiguous_cond = FormatContiguous(cond, at::MemoryFormat::Contiguous);
  Tensor contiguous_self = FormatContiguous(self, at::MemoryFormat::Contiguous);
  Tensor contiguous_other =
      FormatContiguous(other, at::MemoryFormat::Contiguous);

  // 1. deal with other and self tensor shape isn't same
  if (other.dim() == 0) {
    contiguous_other = at::full_like(contiguous_self, other.item());
  }
  if (self.dim() == 0) {
    contiguous_self = at::full_like(contiguous_other, self.item());
  }

  TernaryCall(
      contiguous_out,
      contiguous_cond,
      contiguous_self,
      contiguous_other,
      m,
      alpha_scalar);
  return output;
}

Tensor& WhereSelfOut(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  c10::musa::MUSAGuard device_guard(self.device());
  // check if shapes can be broadcastable and compute output shape
  DimVector output_shape;
  std::vector<std::vector<int64_t>> operands_shape = {
      condition.sizes().vec(), self.sizes().vec(), other.sizes().vec()};
  for (const auto& shape : operands_shape) {
    if (output_shape.empty()) {
      output_shape = DimVector(shape.begin(), shape.end());
    }
    if (output_shape != DimVector(shape.begin(), shape.end())) {
      output_shape = infer_size_dimvector(output_shape, shape);
    }
  }

  Tensor contiguous_self, contiguous_other;
  auto result_type = at::native::result_type(self, other);
  if (self.dtype() != other.dtype()) {
    contiguous_self = self.to(result_type).contiguous();
    contiguous_other = other.to(result_type).contiguous();
  } else {
    contiguous_self = self.contiguous();
    contiguous_other = other.contiguous();
  }
  if (condition.scalar_type() == ScalarType::Byte) {
    TORCH_WARN_ONCE(
        "where received a uint8 condition tensor. This behavior is deprecated "
        "and will be removed in a future version of PyTorch. Use a boolean "
        "condition instead.");
  } else {
    TORCH_CHECK(
        condition.scalar_type() == ScalarType::Bool,
        "where expected condition to be a boolean tensor, but got a "
        "tensor with dtype ",
        condition.scalar_type());
  }
  Tensor cond_bool = condition.scalar_type() == ScalarType::Byte
      ? condition.to(ScalarType::Bool)
      : condition;

  // Keep self, other and condition's shape consistent
  int64_t max_dim = std::max(contiguous_self.dim(), contiguous_other.dim());
  max_dim = std::max(max_dim, cond_bool.dim());

  std::vector<int64_t> expanded_sizes_self(max_dim, 1);
  std::vector<int64_t> expanded_sizes_other(max_dim, 1);
  std::vector<int64_t> expanded_sizes_condition(max_dim, 1);

  int self_len = contiguous_self.dim();
  int other_len = contiguous_other.dim();
  int cond_len = cond_bool.dim();
  for (int i = 0; i < max_dim; ++i) {
    if (i < self_len) {
      expanded_sizes_self[max_dim - 1 - i] =
          contiguous_self.sizes()[self_len - 1 - i];
    }
    if (i < other_len) {
      expanded_sizes_other[max_dim - 1 - i] =
          contiguous_other.sizes()[other_len - 1 - i];
    }
    if (i < cond_len) {
      expanded_sizes_condition[max_dim - 1 - i] =
          cond_bool.sizes()[cond_len - 1 - i];
    }
  }

  if (!out.sizes().equals(output_shape)) {
    out.resize_(output_shape);
  }

  if (!out.numel()) {
    return out;
  }

  // Manually broadcast tensor to match the maximum dimensions, 1.add new
  // axis 2.expand
  Tensor broadcasted_self =
      contiguous_self.reshape(expanded_sizes_self).expand(output_shape);
  Tensor broadcasted_other =
      contiguous_other.reshape(expanded_sizes_other).expand(output_shape);
  Tensor broadcasted_condition =
      cond_bool.reshape(expanded_sizes_condition).expand(output_shape);

  Tensor contiguous_out = out.contiguous();
  return TernaryOut(
      out,
      broadcasted_condition,
      broadcasted_self,
      broadcasted_other,
      TERNARY_MODE::SELECT,
      1);
}

Tensor WhereSelf(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  c10::musa::MUSAGuard device_guard(self.device());
  auto result_type = at::native::result_type(self, other);
  Tensor output = at::empty(
      other.sizes(),
      self.options()
          .dtype(result_type)
          .memory_format(at::MemoryFormat::Contiguous));
  WhereSelfOut(condition, self, other, output);
  return output;
}

Tensor& AddcMulOut(
    const Tensor& self,
    const Tensor& input1,
    const Tensor& input2,
    const Scalar& alpha_scalar,
    Tensor& output) {
  c10::musa::MUSAGuard device_guard(self.device());
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of addcmul must be MUSA, but now it is ",
      self.device());
  TORCH_CHECK(
      input1.device().type() == kMUSA,
      "Device of input1 tensor of addcmul must be MUSA, but now it is ",
      input1.device());
  TORCH_CHECK(
      input2.device().type() == kMUSA,
      "Device of input2 tensor of addcmul must be MUSA, but now it is ",
      input2.device());
  TORCH_CHECK(
      output.device().type() == kMUSA,
      "Device of output tensor of addcmul must be MUSA, but now it is ",
      output.device());
  TernarycommonDtypeCall(
      self, input1, input2, alpha_scalar, output, TERNARY_MODE::ADDCMUL);
  return output;
}

Tensor& AddcDivOut(
    const Tensor& self,
    const Tensor& input1,
    const Tensor& input2,
    const Scalar& alpha_scalar,
    Tensor& output) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of addcdiv must be MUSA, but now it is ",
      self.device());
  TORCH_CHECK(
      input1.device().type() == kMUSA,
      "Device of input1 tensor of addcdiv must be MUSA, but now it is ",
      self.device());
  TORCH_CHECK(
      input2.device().type() == kMUSA,
      "Device of input2 tensor of addcdiv must be MUSA, but now it is ",
      input2.device());
  TORCH_CHECK(
      output.device().type() == kMUSA,
      "Device of output tensor of addcdiv must be MUSA, but now it is ",
      output.device());
  c10::musa::MUSAGuard device_guard(self.device());

  TORCH_CHECK(
      self.scalar_type() == input1.scalar_type() &&
          self.scalar_type() == input2.scalar_type() &&
          self.scalar_type() == output.scalar_type(),
      "Dtype of self, input1, input2, output should be the same");
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Half ||
          self.scalar_type() == at::ScalarType::Int ||
          self.scalar_type() == at::ScalarType::Long ||
          self.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of input tensor of addcdiv only support int32/64, fp32/fp16/bf16, but now it is ",
      self.scalar_type());
  TernarycommonDtypeCall(
      self, input1, input2, alpha_scalar, output, TERNARY_MODE::ADDCDIV);

  return output;
}

at::Tensor& AddcDiv_(
    at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value) {
  // TODO(@mt-ai,mt-compute): Since we don't know the self is created from empty
  // or computed from other operations, and muDNN doesn't supoprt inplace
  // ternary, we create an output then copy it to self
  at::Tensor output = at::empty_like(
      self, self.options().memory_format(at::MemoryFormat::Contiguous));

  AddcDivOut(self, tensor1, tensor2, value, output);

  self.copy_(output);
  return self;
}

at::Tensor AddcDiv(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value) {
  at::TensorIterator iter;
  iter.build_ternary_op(at::Tensor(), self, tensor1, tensor2);
  at::Tensor output = iter.output();
  AddcDivOut(self, tensor1, tensor2, value, output);

  return output;
}

} // namespace musa
} // namespace at
