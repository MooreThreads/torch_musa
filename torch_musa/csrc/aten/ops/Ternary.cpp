#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/BinaryOps.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addcdiv_native.h>
#include <ATen/ops/addcmul_native.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/result_type_native.h>
#include <ATen/ops/where_native.h>
#endif

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

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

  auto contiguous_input1 = input1.contiguous();
  auto contiguous_input2 = input2.contiguous();
  auto contiguous_self = self.contiguous();
  auto input1_mt = CreateMUTensor(contiguous_input1);
  auto input2_mt = CreateMUTensor(contiguous_input2);
  auto self_mt = CreateMUTensor(contiguous_self);
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
  output = output.contiguous();
  Tensor contiguous_self = self.to(common_dtype).contiguous();
  Tensor contiguous_input1 = input1.to(common_dtype).contiguous();
  Tensor contiguous_input2 = input2.to(common_dtype).contiguous();
  TernaryCall(
      output,
      contiguous_self,
      contiguous_input1,
      contiguous_input2,
      m,
      alpha_scalar);
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

  Tensor contiguous_out = output.contiguous();
  Tensor contiguous_cond = cond.contiguous();
  Tensor contiguous_self = self.contiguous();
  Tensor contiguous_other = other.contiguous();

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
  Tensor contiguous_self, contiguous_other;
  auto result_type = at::native::result_type(self, other);
  if (self.dtype() != other.dtype()) {
    contiguous_self = self.to(result_type);
    contiguous_other = other.to(result_type);
  } else {
    contiguous_self = self;
    contiguous_other = other;
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
  // compute output shape
  std::vector<int64_t> condition_shape = condition.sizes().vec();
  DimVector output_shape =
      DimVector(condition_shape.begin(), condition_shape.end());
  if (!out.sizes().equals(output_shape)) {
    out.resize_(output_shape);
  }
  if (!out.numel()) {
    return out;
  }
  Tensor contiguous_out = out.contiguous();

  if (other.dim() == 0) {
    contiguous_other = at::full_like(contiguous_out, contiguous_other.item());
  }
  if (self.dim() == 0) {
    contiguous_self = at::full_like(contiguous_out, contiguous_self.item());
  }
  // we should keep self, other and out's shape consistent
  return TernaryOut(
      out,
      cond_bool,
      contiguous_self,
      contiguous_other,
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
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Half ||
          self.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of input tensor of addcdiv only support fp32/fp16/bf16, but now it is ",
      self.scalar_type());
  TORCH_CHECK(
      input1.scalar_type() == at::ScalarType::Float ||
          input1.scalar_type() == at::ScalarType::Half ||
          input1.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of input1 tensor of addcdiv only support fp32/fp16/bf16, but now it is ",
      input1.scalar_type());
  TORCH_CHECK(
      input2.scalar_type() == at::ScalarType::Float ||
          input2.scalar_type() == at::ScalarType::Half ||
          input2.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of input2 tensor of addcdiv only support fp32/fp16/bf16 but now it is ",
      input2.scalar_type());
  TORCH_CHECK(
      output.dtype() == at::ScalarType::Float ||
          output.dtype() == at::ScalarType::Half ||
          output.dtype() == at::ScalarType::BFloat16,
      "Dtype of output tensor of addcdiv only support fp32/fp16/bf16, but now it is ",
      output.dtype());
  TernarycommonDtypeCall(
      self, input1, input2, alpha_scalar, output, TERNARY_MODE::ADDCDIV);
  return output;
}

ADVANCED_REGISTER(aten, PrivateUse1, "where.self", WhereSelf)
ADVANCED_REGISTER(aten, PrivateUse1, "where.self_out", WhereSelfOut)

ADVANCED_REGISTER(aten, PrivateUse1, "addcdiv.out", AddcDivOut)
ADVANCED_REGISTER(aten, PrivateUse1, "addcmul.out", AddcMulOut)

} // namespace musa
} // namespace at
