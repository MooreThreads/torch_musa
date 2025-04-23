#include <ATen/Config.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/BinaryOps.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_to_copy.h>
#include <ATen/ops/add.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/and_native.h>
#include <ATen/ops/arctan2_native.h>
#include <ATen/ops/div.h>
#include <ATen/ops/div_native.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/divide_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/eq_native.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/floor_divide_native.h>
#include <ATen/ops/fmax_native.h>
#include <ATen/ops/fmin_native.h>
#include <ATen/ops/full.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/ge_native.h>
#include <ATen/ops/greater_equal_native.h>
#include <ATen/ops/greater_native.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/gt_native.h>
#include <ATen/ops/hardswish_backward_native.h>
#include <ATen/ops/heaviside_native.h>
#include <ATen/ops/hypot_native.h>
#include <ATen/ops/le.h>
#include <ATen/ops/le_native.h>
#include <ATen/ops/less_equal_native.h>
#include <ATen/ops/less_native.h>
#include <ATen/ops/linalg_cross_native.h>
#include <ATen/ops/linalg_cross_ops.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/lt_native.h>
#include <ATen/ops/max_native.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/maximum_native.h>
#include <ATen/ops/min_native.h>
#include <ATen/ops/minimum.h>
#include <ATen/ops/minimum_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/mul_ops.h>
#include <ATen/ops/multiply_native.h>
#include <ATen/ops/remainder.h>
#include <ATen/ops/remainder_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/sigmoid_backward_native.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sub_native.h>
#include <ATen/ops/subtract_native.h>
#include <ATen/ops/tanh_backward_native.h>
#endif

#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {
using BINARY_MODE = ::musa::dnn::Binary::Mode;
using UNARY_MODE = ::musa::dnn::Unary::Mode;

inline bool IsComparisonOp(const BINARY_MODE m) {
  return m == BINARY_MODE::EQ || m == BINARY_MODE::NE || m == BINARY_MODE::GE ||
      m == BINARY_MODE::GT || m == BINARY_MODE::LE || m == BINARY_MODE::LT;
}

// only supports when muDNN has relate UNARY_MODE
inline bool SupportMode(BINARY_MODE m) {
  return (
      m == BINARY_MODE::ADD || m == BINARY_MODE::SUB || m == BINARY_MODE::MUL ||
      m == BINARY_MODE::FLOORMOD || m == BINARY_MODE::TRUEDIV ||
      m == BINARY_MODE::FLOORDIV || m == BINARY_MODE::TRUNCATEDIV);
}

inline bool SupportType(at::ScalarType type) {
  return type == ScalarType::Float || type == ScalarType::Half ||
      type == ScalarType::Int || type == ScalarType::Long ||
      type == ScalarType::BFloat16;
}

inline bool SupportOptimizeScalarToUnary(
    BINARY_MODE m,
    const Tensor& input,
    const Tensor& other) {
  const bool support_mode = SupportMode(m);
  const bool support_type = SupportType(input.scalar_type());
  const bool support_dim_device =
      other.dim() == 0 && other.device() == DeviceType::CPU;
  return support_mode && support_type && support_dim_device;
};

inline bool IsBoolMode(BINARY_MODE m) {
  if (m == BINARY_MODE::EQ || m == BINARY_MODE::NE || m == BINARY_MODE::GE ||
      m == BINARY_MODE::GT || m == BINARY_MODE::LE || m == BINARY_MODE::LT ||
      m == BINARY_MODE::LOGICAL_AND || m == BINARY_MODE::LOGICAL_OR ||
      m == BINARY_MODE::LOGICAL_XOR) {
    return true;
  } else {
    return false;
  }
}

inline bool is_scalar(const at::Tensor& tensor) {
  return tensor.numel() == 1 && tensor.dim() == 0;
}

void UnaryCall(
    const Tensor& self,
    const Tensor& other,
    Tensor& output,
    BINARY_MODE m,
    const std::string op_name) {
  if (C10_UNLIKELY(self.numel() == 0 || other.numel() == 0)) {
    return;
  }
  bool is_other_integer = false;
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Unary uop;
  auto other_scalar = other.item();
  if (other_scalar.isFloatingPoint()) {
    CHECK_MUDNN_STATUS(uop.SetAlpha(other_scalar.toDouble()), "SetAlpha");
  } else if (other_scalar.isIntegral(false)) {
    is_other_integer = true;
    CHECK_MUDNN_STATUS(uop.SetAlpha(other_scalar.toLong()), "SetAlpha");
  } else {
    AT_ERROR(
        other_scalar.type(), " is not implemented for broadcast in Binary");
  }

  if (self.numel() == 0) {
    return;
  }

  if (m == BINARY_MODE::MUL) {
    CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::MUL), "SetMode");
  } else if (m == BINARY_MODE::ADD) {
    CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::ADD), "SetMode");
  } else if (m == BINARY_MODE::TRUEDIV) {
    // truediv with integer input and integer scalar divider should output a
    // fp32 tensor instead of keeping the dtype
    output = is_other_integer &&
            (self.scalar_type() == at::ScalarType::Int ||
             self.scalar_type() == at::ScalarType::Long)
        ? output.to(at::ScalarType::Float)
        : output;
    CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::TRUEDIV), "SetMode");
  } else if (m == BINARY_MODE::SUB) {
    CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::SUB), "SetMode");
  } else if (m == BINARY_MODE::FLOORMOD) {
    CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::FLOORMOD), "SetMode");
  } else if (m == BINARY_MODE::FLOORDIV) {
    CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::FLOORDIV), "SetMode");
  } else if (m == BINARY_MODE::TRUNCATEDIV) {
    CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::TRUNCATEDIV), "SetMode");
  } else {
    AT_ERROR("Invalid mode for broadcast in Binary");
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      output.suggest_memory_format() == self.suggest_memory_format());
  auto mt_input = CreateMUTensor(self);
  auto mt_output = CreateMUTensor(output);
  CHECK_MUDNN_STATUS(uop.Run(h, mt_output, mt_input), "Run " + op_name);
}

/**
 * @brief Binary ops calling convention.
 *
 * @param op_name Binary OP's name
 * @param output output tensor to write result， this tensor could be
 * non-contiguous.
 * @param self first operand, must be contiguous passed by caller
 * @param other second operand, must be contiguous passed by caller
 * @param m Binary Mode Type.
 * @param alpha_scalar scaling factor.
 */
void BinaryCall(
    const std::string& op_name,
    Tensor& output,
    const Tensor& self,
    const Tensor& other,
    BINARY_MODE m = BINARY_MODE::ADD,
    Scalar const& alpha_scalar = 1) {
  if (C10_UNLIKELY(self.numel() == 0 || other.numel() == 0)) {
    return;
  }
  Device device = is_musa(self) ? self.device() : other.device();
  const c10::musa::MUSAGuard guard(device);

  // There are only two types of inputs for the binary operator: musa tensor and
  // CPU scalar, or two musa tensors. So when one of the Tensors is on the CPU
  // and not a scalar, it means there is a problem
  if ((!is_scalar(other) && other.device().is_cpu()) ||
      (!is_scalar(self) && self.device().is_cpu())) {
    TORCH_CHECK(
        false,
        "Expected all tensors to be on the same device, but "
        "found at least two devices, ",
        self.device().type(),
        " and ",
        other.device().type(),
        "!")
  };
  if (is_musa(self) && is_musa(other) &&
      self.device().index() != other.device().index()) {
    TORCH_CHECK(
        false,
        "Expected all tensors to be on the same device, but "
        "found at least two devices, ",
        self.device(),
        " and ",
        other.device(),
        "!")
  };

  muHandle& h = GetMudnnHandle();
  if (self.numel() == 0 && other.numel() == 0) {
    Tensor out_tmp;
    if (IsComparisonOp(m)) {
      out_tmp = at::empty(
          output.sizes(),
          self.options().dtype(ScalarType::Bool),
          output.suggest_memory_format());
    } else {
      out_tmp = at::empty(
          output.sizes(), self.options(), output.suggest_memory_format());
    }
    if (output.numel() > 0) {
      output.copy_(out_tmp);
    }
    return;
  }

  if (SupportOptimizeScalarToUnary(m, self, other)) {
    Tensor other_ =
        alpha_scalar.equal(1) ? other : at::mul(other, alpha_scalar);
    UnaryCall(self, other_, output, m, op_name);
    return;
  }

  if (SupportOptimizeScalarToUnary(m, other, self)) {
    Tensor other_ =
        alpha_scalar.equal(1) ? other : at::mul(other, alpha_scalar);
    if (m == BINARY_MODE::ADD || m == BINARY_MODE::MUL) {
      UnaryCall(other_, self, output, m, op_name);
      return;
    }
  }

  Tensor self_tmp = self;
  Tensor other_tmp = other;

  if (self_tmp.dim() == 0 && other_tmp.dim() != 0 && m != BINARY_MODE::ADD &&
      m != BINARY_MODE::MUL) {
    self_tmp = at::full_like(other, self_tmp.item(), kMUSA);
  }
  if (!is_musa(self_tmp)) {
    self_tmp = self.to(device);
  }
  if (!is_musa(other_tmp)) {
    other_tmp = other.to(device);
  }
  other_tmp =
      alpha_scalar.equal(1) ? other_tmp : at::mul(other_tmp, alpha_scalar);
  Tensor self_;
  Tensor other_;

  const auto out_memory_format = output.suggest_memory_format();

  if (self_tmp.suggest_memory_format() != out_memory_format) {
    self_ = FormatContiguous(self_tmp, out_memory_format);
  } else {
    self_ = self_tmp;
  }
  if (other_tmp.suggest_memory_format() != out_memory_format) {
    other_ = FormatContiguous(other_tmp, out_memory_format);
  } else {
    other_ = other_tmp;
  }
  // TODO(kang.chen): mudnn not support int, cast to float temporarily.
  bool is_trans = false;
  ScalarType out_dtype = at::ScalarType::Long;
  if (m == BINARY_MODE::POW &&
      at::isIntegralType(self_tmp.scalar_type(), false) &&
      at::isIntegralType(other_tmp.scalar_type(), false)) {
    is_trans = true;
    out_dtype = output.scalar_type();
    self_ = self_tmp.to(ScalarType::Float);
    other_ = other_tmp.to(ScalarType::Float);
    output = output.to(ScalarType::Float);
  }
  muTensor musa_self = CreateMUTensor(self_);
  muTensor musa_other = CreateMUTensor(other_);
  muTensor musa_out = CreateMUTensor(output);

  ::musa::dnn::Binary bop;
  CHECK_MUDNN_STATUS(bop.SetMode(m), "SetMode");
  CHECK_MUDNN_STATUS(
      bop.Run(h, musa_out, musa_self, musa_other), "Run " + op_name);
  if (is_trans) {
    output = output.to(out_dtype);
  }
}

extern Tensor create_out(
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options);

extern void check_inplace(
    const Tensor& self,
    IntArrayRef sizes,
    const TensorOptions& options);

Tensor Binary(
    const std::string op_name,
    const Tensor& self,
    const Tensor& other,
    BINARY_MODE m = BINARY_MODE::ADD,
    Scalar const& alpha_scalar = 1) {
  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "input scalar type must the same");

  // One of the both tensors might be cpu device. e.g.
  // 1. self will be in cpu if '1 + Tensor'.
  // 2. other wiil be in cpu if 'Tensor + 1'.
  // We use get musa devcie info to set context, so we need this check.
  Device device = is_musa(self) ? self.device() : other.device();
  c10::musa::MUSAGuard guard(device);
  Tensor output;
  auto output_sizes = infer_size_dimvector(self.sizes(), other.sizes());
  TORCH_CHECK(
      is_expandable_to(other.sizes(), output_sizes),
      "size {",
      self.sizes(),
      "} is not expandable to size {",
      output_sizes,
      "}.");
  TORCH_CHECK(
      is_expandable_to(self.sizes(), output_sizes),
      "size {",
      self.sizes(),
      "} is not expandable to size {",
      output_sizes,
      "}.");
  auto output_dtype = self.scalar_type();
  if (IsBoolMode(m)) {
    output_dtype = ScalarType::Bool;
  }
  auto output_memory_format = self.suggest_memory_format();
  if (output_memory_format == at::MemoryFormat::Contiguous) {
    output_memory_format = other.suggest_memory_format();
  }
  output = at::empty(
      output_sizes,
      self.options()
          .dtype(output_dtype)
          .device(device)
          .memory_format(output_memory_format));
  BinaryCall(op_name, output, self, other, m, alpha_scalar);
  return output;
}

Tensor BinarycommonDtype(
    const std::string& op_name,
    const Tensor& self,
    const Tensor& other,
    Scalar const& alpha_scalar,
    BINARY_MODE m) {
  Device device = is_musa(self) ? self.device() : other.device();
  c10::musa::MUSAGuard device_guard(device);
  if ((self.scalar_type() == ScalarType::Bool &&
       other.scalar_type() == ScalarType::Bool) ||
      (self.scalar_type() == ScalarType::Double &&
       other.scalar_type() == ScalarType::Double)) {
    if (m == BINARY_MODE::MUL) {
      return at::mul(self.cpu(), other.cpu()).to(device);
    } else if (m == BINARY_MODE::TRUEDIV) {
      return at::div(self.cpu(), other.cpu()).to(device);
    }
  }
  ScalarType common_dtype = at::result_type(self, other);
  at::native::alpha_check(common_dtype, alpha_scalar);
  Tensor common_self =
      at::musa::ContiguousIfZeroInStrides(self.to(common_dtype));
  Tensor common_other =
      at::musa::ContiguousIfZeroInStrides(other.to(common_dtype));
  return Binary(op_name, common_self, common_other, m, alpha_scalar);
}

void BinarycommonDtypeCall(
    const std::string& op_name,
    const Tensor& self,
    const Tensor& other,
    Scalar const& alpha_scalar,
    Tensor& output,
    BINARY_MODE m) {
  ScalarType common_dtype = at::result_type(self, other);
  at::native::alpha_check(common_dtype, alpha_scalar);
  Tensor common_self = self.to(common_dtype);
  Tensor common_other = other.to(common_dtype);
  auto out_type = IsBoolMode(m) ? ScalarType::Bool : common_dtype;
  if (output.scalar_type() == out_type) {
    BinaryCall(op_name, output, common_self, common_other, m, alpha_scalar);
  } else {
    auto common_output = output.to(out_type);
    BinaryCall(
        op_name, common_output, common_self, common_other, m, alpha_scalar);
    output.copy_(common_output);
  }
}

#define DEFINE_BINARY_OP(op_name, mode)                             \
  Tensor op_name##Tensor(const Tensor& self, const Tensor& other) { \
    return BinarycommonDtype(__func__, self, other, 1, mode);       \
  }                                                                 \
                                                                    \
  Tensor& op_name##Tensor_(Tensor& self, const Tensor& other) {     \
    BinarycommonDtypeCall(__func__, self, other, 1, self, mode);    \
    return self;                                                    \
  }                                                                 \
                                                                    \
  Tensor& op_name##TensorOut(                                       \
      const Tensor& self, const Tensor& other, Tensor& output) {    \
    BinarycommonDtypeCall(__func__, self, other, 1, output, mode);  \
    return output;                                                  \
  }

DEFINE_BINARY_OP(Equal, BINARY_MODE::EQ)
DEFINE_BINARY_OP(NotEqual, BINARY_MODE::NE)
DEFINE_BINARY_OP(Greater, BINARY_MODE::GT)
DEFINE_BINARY_OP(GreaterEqual, BINARY_MODE::GE)
DEFINE_BINARY_OP(LessEqual, BINARY_MODE::LE)
DEFINE_BINARY_OP(Remainder, BINARY_MODE::FLOORMOD)
DEFINE_BINARY_OP(Less, BINARY_MODE::LT)
DEFINE_BINARY_OP(Maximum, BINARY_MODE::MAX)
DEFINE_BINARY_OP(Minimum, BINARY_MODE::MIN)
DEFINE_BINARY_OP(LogicalAnd, BINARY_MODE::LOGICAL_AND)
DEFINE_BINARY_OP(LogicalOr, BINARY_MODE::LOGICAL_OR)
DEFINE_BINARY_OP(LogicalXor, BINARY_MODE::LOGICAL_XOR)
DEFINE_BINARY_OP(Pow, BINARY_MODE::POW)

Tensor RemainderScalarTensor(const Scalar& self, const Tensor& other) {
  auto self_tmp = at::full_like(other, self, kMUSA);
  return at::remainder(self_tmp, other);
}

#define DEFINE_BINARY_GRAD_OP(op_name, mode)                                 \
  Tensor op_name(const Tensor& grad_output, const Tensor& self) {            \
    return BinarycommonDtype(__func__, grad_output, self, 1, mode);          \
  }                                                                          \
                                                                             \
  Tensor& op_name##Out(                                                      \
      const Tensor& grad_output, const Tensor& self, Tensor& grad_input) {   \
    BinarycommonDtypeCall(__func__, grad_output, self, 1, grad_input, mode); \
    return grad_input;                                                       \
  }

DEFINE_BINARY_GRAD_OP(SiluBwd, BINARY_MODE::SILU_BW)
DEFINE_BINARY_GRAD_OP(SigmoidBwd, BINARY_MODE::SIGMOID_BW)
DEFINE_BINARY_GRAD_OP(TanhBwd, BINARY_MODE::TANH_BW)

at::Tensor& GeluBwdOut(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    c10::string_view approximate,
    at::Tensor& grad_input) {
  const c10::musa::MUSAGuard device_guard(self.device());
  const auto memory_format = grad_input.suggest_memory_format();
  Tensor contiguous_grad_output = FormatContiguous(grad_output, memory_format);
  Tensor contiguous_self = FormatContiguous(self, memory_format);

  grad_input.resize_(self.sizes());
  auto approximate_type = at::native::get_gelutype_enum(approximate);
  auto mode = approximate_type == at::native::GeluType::None
      ? ::musa::dnn::Binary::Mode::GELU_NONE_BW
      : ::musa::dnn::Binary::Mode::GELU_TANH_BW;
  BinaryCall(
      __func__, grad_input, contiguous_grad_output, contiguous_self, mode);

  return grad_input;
}

at::Tensor GeluBwd(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    c10::string_view approximate) {
  const c10::musa::MUSAGuard device_guard(self.device());
  auto result =
      ::at::empty(self.sizes(), self.options(), self.suggest_memory_format());
  GeluBwdOut(grad_output, self, approximate, result);
  return result;
}

// TODO(zaixing.wang): maybe we can use macro to refactor unary/binary op with
// special args like Threadshold, GELU, PowScalar.
Tensor& ThresholdBwdOut(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold,
    Tensor& grad_input) {
  if (grad_output.numel() == 0 && self.numel() == 0)
    return grad_input;
  const c10::musa::MUSAGuard device_gaurd(self.device());
  const auto grad_input_memory_format = grad_input.suggest_memory_format();
  auto contiguous_grad_output =
      FormatContiguous(grad_output, grad_input_memory_format);
  auto contiguous_self = FormatContiguous(self, grad_input_memory_format);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Binary binary_op;
  auto mt_grad_output = CreateMUTensor(contiguous_grad_output);
  auto mt_self = CreateMUTensor(contiguous_self);
  auto mt_output = CreateMUTensor(grad_input);
  CHECK_MUDNN_STATUS(
      binary_op.SetMode(::musa::dnn::Binary::Mode::THRESHOLD_BW), "SetMode");
  // only support float32 now
  AT_DISPATCH_ALL_MTGPU_TYPES_AND_HALF(
      self.scalar_type(), "threshold_backward", [&] {
        auto threshold_value = threshold.to<scalar_t>();
        if (self.scalar_type() == at::ScalarType::Double ||
            self.scalar_type() == at::ScalarType::Float ||
            self.scalar_type() == at::ScalarType::Half ||
            self.scalar_type() == at::ScalarType::BFloat16) {
          CHECK_MUDNN_STATUS(
              binary_op.SetAlpha(static_cast<double>(threshold_value)),
              "SetAlpha");
        } else {
          CHECK_MUDNN_STATUS(
              binary_op.SetAlpha(static_cast<int64_t>(threshold_value)),
              "SetAlpha");
        }
        CHECK_MUDNN_STATUS(
            binary_op.Run(h, mt_output, mt_grad_output, mt_self), "Run");
      });
  return grad_input;
}

Tensor ThresholdBwd(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold) {
  Tensor grad_input =
      at::empty(self.sizes(), self.options(), self.suggest_memory_format());
  ThresholdBwdOut(grad_output, self, threshold, grad_input);
  return grad_input;
}

bool MUSAEqual(const at::Tensor& self, const at::Tensor& other) {
  c10::musa::MUSAGuard device_guard(self.device());
  TORCH_CHECK(
      self.device() == other.device(),
      "Cannot compare two tensors on "
      "different devices. Got: ",
      self.device(),
      " and ",
      other.device());
  if (self.sizes() != other.sizes()) {
    return false;
  }
  if (self.numel() == 0) {
    return true;
  }
  return at::musa::EqualTensor(self, other).all().item().to<bool>();
}

} // namespace musa
} // namespace at
