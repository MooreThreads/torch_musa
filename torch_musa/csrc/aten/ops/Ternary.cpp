#include <ATen/ATen.h>
#include <ATen/native/BinaryOps.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace native {
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

  auto input1_mt = CreateMUTensor(input1);
  auto input2_mt = CreateMUTensor(input2);
  auto self_mt = CreateMUTensor(self);
  auto om_mt = CreateMUTensor(output);
  CHECK_MUDNN_STATUS(top.SetMode(m), "SetMode");
  CHECK_MUDNN_STATUS(top.Run(h, om_mt, self_mt, input1_mt, input2_mt), "Run");
}

void TernarycommonDtypeCall(
    const Tensor& self,
    const Tensor& input1,
    const Tensor& input2,
    Scalar const& alpha_scalar,
    Tensor& output,
    TERNARY_MODE m) {
  auto common_dtype = at::result_type(input1, input2);
  alpha_check(common_dtype, alpha_scalar);
  Tensor contiguous_self = Contiguous(self.to(common_dtype));
  Tensor contiguous_input1 = Contiguous(input1.to(common_dtype));
  Tensor contiguous_input2 = Contiguous(input2.to(common_dtype));
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

  Tensor contiguous_out = Contiguous(output);
  Tensor contiguous_cond = Contiguous(cond);
  Tensor contiguous_self = Contiguous(self);
  Tensor contiguous_other = Contiguous(other);

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

  // TODO(caizhi): using "out.resize_()" to replace "empty_mtgpu" would be
  // better, but now memcpyD2D(*dst, *src) function is not supported in muDNN
  // invoking "out.resize_()", which may be supported in muDNN.
  if (!out.sizes().equals(output_shape)) {
    out = empty_mtgpu(
        output_shape,
        result_type,
        c10::nullopt,
        self.device(),
        c10::nullopt,
        at::MemoryFormat::Contiguous);
  }
  if (!out.numel()) {
    return out;
  }
  return TernaryOut(out, cond_bool, self, other, TERNARY_MODE::SELECT, 1);
}

Tensor WhereSelf(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  auto result_type = at::native::result_type(self, other);
  Tensor output = empty_mtgpu(
      other.sizes(),
      result_type,
      c10::nullopt,
      self.device(),
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  WhereSelfOut(condition, self, other, output);
  return output;
}

Tensor& AddcMulOut(
    const Tensor& self,
    const Tensor& input1,
    const Tensor& input2,
    const Scalar& alpha_scalar,
    Tensor& output) {
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
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of addcdiv only support Float32, but now it is ",
      self.scalar_type());
  TORCH_CHECK(
      input1.scalar_type() == at::ScalarType::Float,
      "Dtype of input1 tensor of addcdiv only support Float32, but now it is ",
      input1.scalar_type());
  TORCH_CHECK(
      input2.scalar_type() == at::ScalarType::Float,
      "Dtype of input2 tensor of addcdiv only support Float32, but now it is ",
      input2.scalar_type());
  TORCH_CHECK(
      output.dtype() == at::ScalarType::Float,
      "Dtype of output tensor of addcdiv only support Float32, but now it is ",
      output.dtype());
  TernarycommonDtypeCall(
      self, input1, input2, alpha_scalar, output, TERNARY_MODE::ADDCDIV);
  return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("where.self", &WhereSelf);
  m.impl("where.self_out", &WhereSelfOut);

  m.impl("addcdiv.out", &AddcDivOut);
  m.impl("addcmul.out", &AddcMulOut);
}

} // namespace musa
} // namespace native
} // namespace at
