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
#include <ATen/ops/atan2.h>
#include <ATen/ops/atan2_native.h>
#include <ATen/ops/bitwise_and.h>
#include <ATen/ops/bitwise_and_native.h>
#include <ATen/ops/bitwise_or.h>
#include <ATen/ops/bitwise_or_native.h>
#include <ATen/ops/bitwise_right_shift.h>
#include <ATen/ops/bitwise_right_shift_native.h>
#include <ATen/ops/bitwise_xor.h>
#include <ATen/ops/bitwise_xor_native.h>
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
#include <ATen/ops/xlogy.h>
#include <ATen/ops/xlogy_native.h>
#endif

#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

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
      type == ScalarType::Int || type == ScalarType::Long;
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
  } else if (m == BINARY_MODE::FLOORDIV) {
    CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::FLOORDIV), "SetMode");
  } else {
    AT_ERROR("Invalid mode for broadcast in Binary");
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      output.suggest_memory_format() == self.suggest_memory_format());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      output.is_contiguous(self.suggest_memory_format()));
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
    UnaryCall(self, other, output, m, op_name);
    return;
  }

  if (SupportOptimizeScalarToUnary(m, other, self)) {
    if (m == BINARY_MODE::ADD || m == BINARY_MODE::MUL) {
      UnaryCall(other, self, output, m, op_name);
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
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(output.is_contiguous(out_memory_format));

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

  muTensor musa_self = CreateMUTensor(self_);
  muTensor musa_other = CreateMUTensor(other_);
  muTensor musa_out = CreateMUTensor(output);

  ::musa::dnn::Binary bop;
  CHECK_MUDNN_STATUS(bop.SetMode(m), "SetMode");
  CHECK_MUDNN_STATUS(
      bop.Run(h, musa_out, musa_self, musa_other), "Run " + op_name);
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

// TODO(mt-ai): All the binary operations should be moved to the musa
// implementation and compiled using mcc.
#define DEFINE_BINARY_SCALAR_OP(op_name, mode)                                \
  Tensor op_name##Tensor(                                                     \
      const Tensor& self, const Tensor& other, Scalar const& alpha_scalar) {  \
    return BinarycommonDtype(__func__, self, other, alpha_scalar, mode);      \
  }                                                                           \
                                                                              \
  Tensor& op_name##_Tensor(                                                   \
      Tensor& self, const Tensor& other, Scalar const& alpha_scalar) {        \
    BinarycommonDtypeCall(__func__, self, other, alpha_scalar, self, mode);   \
    return self;                                                              \
  }                                                                           \
                                                                              \
  Tensor& op_name##_out(                                                      \
      const Tensor& self,                                                     \
      const Tensor& other,                                                    \
      Scalar const& alpha_scalar,                                             \
      Tensor& output) {                                                       \
    BinarycommonDtypeCall(__func__, self, other, alpha_scalar, output, mode); \
    return output;                                                            \
  }

DEFINE_BINARY_SCALAR_OP(Add, BINARY_MODE::ADD)
DEFINE_BINARY_SCALAR_OP(Sub, BINARY_MODE::SUB)

#define DEFINE_BINARY_OP(op_name, mode)                             \
  Tensor op_name##Tensor(const Tensor& self, const Tensor& other) { \
    return BinarycommonDtype(__func__, self, other, 1, mode);       \
  }                                                                 \
                                                                    \
  Tensor& op_name##_Tensor(Tensor& self, const Tensor& other) {     \
    BinarycommonDtypeCall(__func__, self, other, 1, self, mode);    \
    return self;                                                    \
  }                                                                 \
                                                                    \
  Tensor& op_name##_out(                                            \
      const Tensor& self, const Tensor& other, Tensor& output) {    \
    BinarycommonDtypeCall(__func__, self, other, 1, output, mode);  \
    return output;                                                  \
  }

DEFINE_BINARY_OP(Mul, BINARY_MODE::MUL)
DEFINE_BINARY_OP(Div, BINARY_MODE::TRUEDIV)
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
DEFINE_BINARY_OP(Pow, BINARY_MODE::POW)
DEFINE_BINARY_OP(FloorDivide, BINARY_MODE::FLOORDIV)

Tensor& Div_out_mode(
    const Tensor& self,
    const Tensor& other,
    c10::optional<c10::string_view> rounding_mode,
    Tensor& output) {
  if (!rounding_mode.has_value()) {
    BinarycommonDtypeCall(
        __func__, self, other, 1, output, BINARY_MODE::TRUEDIV);
  } else if (*rounding_mode == "trunc") {
    BinarycommonDtypeCall(
        __func__, self, other, 1, output, BINARY_MODE::TRUNCATEDIV);
  } else if (*rounding_mode == "floor") {
    BinarycommonDtypeCall(
        __func__, self, other, 1, output, BINARY_MODE::FLOORDIV);
  }
  return output;
}

Tensor DivTensor_mode(
    const Tensor& self,
    const Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  if (!rounding_mode.has_value()) {
    return BinarycommonDtype(__func__, self, other, 1, BINARY_MODE::TRUEDIV);
  } else if (*rounding_mode == "trunc") {
    return BinarycommonDtype(
        __func__, self, other, 1, BINARY_MODE::TRUNCATEDIV);
  } else if (*rounding_mode == "floor") {
    return BinarycommonDtype(__func__, self, other, 1, BINARY_MODE::FLOORDIV);
  } else {
    return self;
  }
}

Tensor& Div_Tensor_mode(
    Tensor& self,
    const Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  if (!rounding_mode.has_value()) {
    BinarycommonDtypeCall(__func__, self, other, 1, self, BINARY_MODE::TRUEDIV);
  } else if (*rounding_mode == "trunc") {
    BinarycommonDtypeCall(
        __func__, self, other, 1, self, BINARY_MODE::TRUNCATEDIV);
  } else if (*rounding_mode == "floor") {
    BinarycommonDtypeCall(
        __func__, self, other, 1, self, BINARY_MODE::FLOORDIV);
  }
  return self;
}

Tensor RemainderScalarTensor(const Scalar& self, const Tensor& other) {
  auto self_tmp = at::full_like(other, self, kMUSA);
  return at::remainder(self_tmp, other);
}

#define DEFINE_BINARY_GRAD_OP(op_name, mode)                                 \
  Tensor op_name(const Tensor& grad_output, const Tensor& self) {            \
    return BinarycommonDtype(__func__, grad_output, self, 1, mode);          \
  }                                                                          \
                                                                             \
  Tensor& op_name##_out(                                                     \
      const Tensor& grad_output, const Tensor& self, Tensor& grad_input) {   \
    BinarycommonDtypeCall(__func__, grad_output, self, 1, grad_input, mode); \
    return grad_input;                                                       \
  }

DEFINE_BINARY_GRAD_OP(SiluBwd, BINARY_MODE::SILU_BW)
DEFINE_BINARY_GRAD_OP(SigmoidBwd, BINARY_MODE::SIGMOID_BW)
DEFINE_BINARY_GRAD_OP(TanhBwd, BINARY_MODE::TANH_BW)

at::Tensor& GELUBwd_out(
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

at::Tensor GELUBwd(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    c10::string_view approximate) {
  const c10::musa::MUSAGuard device_guard(self.device());
  auto result =
      ::at::empty(self.sizes(), self.options(), self.suggest_memory_format());
  GELUBwd_out(grad_output, self, approximate, result);
  return result;
}

// TODO(zaixing.wang): maybe we can use macro to refactor unary/binary op with
// special args like Threadshold, GELU, PowScalar.
Tensor& ThresholdBwd_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold,
    Tensor& grad_input) {
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
            self.scalar_type() == at::ScalarType::Half) {
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
  ThresholdBwd_out(grad_output, self, threshold, grad_input);
  return grad_input;
}

namespace {
#define DEFINE_STRUCTURED_BITWISE_OP_F(op_name)                                \
  struct structured_bitwise_##op_name##_out_functional final                   \
      : public at::native::structured_bitwise_##op_name##_out {                \
    void set_output_strided(                                                   \
        int64_t output_idx,                                                    \
        IntArrayRef sizes,                                                     \
        IntArrayRef strides,                                                   \
        TensorOptions options,                                                 \
        DimnameList names) override {                                          \
      auto current_device = guard_.current_device();                           \
      if (C10_UNLIKELY(current_device.has_value())) {                          \
        TORCH_INTERNAL_ASSERT(                                                 \
            *current_device == options.device(),                               \
            "structured kernels don't support multi-device outputs");          \
      } else {                                                                 \
        guard_.reset_device(options.device());                                 \
      }                                                                        \
      outputs_[output_idx] = create_out(sizes, strides, options);              \
      if (!names.empty()) {                                                    \
        namedinference::propagate_names(*outputs_[output_idx], names);         \
      }                                                                        \
      at::native::structured_bitwise_##op_name##_out ::set_output_raw_strided( \
          output_idx, sizes, strides, options, names);                         \
    }                                                                          \
    void set_output_raw_strided(                                               \
        int64_t output_idx,                                                    \
        IntArrayRef sizes,                                                     \
        IntArrayRef strides,                                                   \
        TensorOptions options,                                                 \
        DimnameList names) override {                                          \
      set_output_strided(output_idx, sizes, strides, options, names);          \
    }                                                                          \
    const Tensor& maybe_get_output(int64_t output_idx) override {              \
      return *outputs_[output_idx];                                            \
    }                                                                          \
    std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;                     \
    c10::musa::OptionalMUSAGuard guard_;                                       \
  };                                                                           \
                                                                               \
  at::Tensor& Bitwise_##op_name##_tensor_out(                                  \
      const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {      \
    structured_bitwise_##op_name##_out_out op(out);                            \
    op.meta(self, other);                                                      \
    op.impl(self, other, op.maybe_get_output(0));                              \
    if (op.proxy_outputs_[0].has_value())                                      \
      op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);                      \
    return out;                                                                \
  }                                                                            \
  at::Tensor& Bitwise_##op_name##_tensor_inplace(                              \
      at::Tensor& self, const at::Tensor& other) {                             \
    structured_bitwise_##op_name##_out_inplace op(self);                       \
    op.meta(self, other);                                                      \
    op.impl(self, other, op.outputs_[0]);                                      \
    if (op.proxy_outputs_[0].has_value())                                      \
      op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);                      \
    return self;                                                               \
  }                                                                            \
  at::Tensor Bitwise_##op_name##_tensor(                                       \
      const at::Tensor& self, const at::Tensor& other) {                       \
    structured_bitwise_##op_name##_out_functional op;                          \
    op.meta(self, other);                                                      \
    op.impl(self, other, *op.outputs_[0]);                                     \
    return std::move(op.outputs_[0]).take();                                   \
  }

#define DEFINE_STRUCTURED_BITWISE_OP_MODE(name, mode)                       \
  struct structured_bitwise_##name##_out_##mode final                       \
      : public at::native::structured_bitwise_##name##_out {                \
    structured_bitwise_##name##_out_##mode(Tensor& self)                    \
        : outputs_{std::ref(self)} {}                                       \
    void set_output_strided(                                                \
        int64_t output_idx,                                                 \
        IntArrayRef sizes,                                                  \
        IntArrayRef strides,                                                \
        TensorOptions options,                                              \
        DimnameList names) override {                                       \
      auto current_device = guard_.current_device();                        \
      if (C10_UNLIKELY(current_device.has_value())) {                       \
        TORCH_INTERNAL_ASSERT(                                              \
            *current_device == options.device(),                            \
            "structured kernels don't support multi-device outputs");       \
      } else {                                                              \
        guard_.reset_device(options.device());                              \
      }                                                                     \
      const auto& out = outputs_[output_idx].get();                         \
      check_inplace(out, sizes, options);                                   \
      auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);  \
      if (C10_UNLIKELY(maybe_proxy.has_value())) {                          \
        proxy_outputs_[output_idx] =                                        \
            c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());  \
      }                                                                     \
      if (!names.empty()) {                                                 \
        namedinference::propagate_names(outputs_[output_idx], names);       \
      }                                                                     \
      at::native::structured_bitwise_##name##_out ::set_output_raw_strided( \
          output_idx, sizes, strides, options, names);                      \
    }                                                                       \
    void set_output_raw_strided(                                            \
        int64_t output_idx,                                                 \
        IntArrayRef sizes,                                                  \
        IntArrayRef strides,                                                \
        TensorOptions options,                                              \
        DimnameList names) override {                                       \
      auto current_device = guard_.current_device();                        \
      if (C10_UNLIKELY(current_device.has_value())) {                       \
        TORCH_INTERNAL_ASSERT(                                              \
            *current_device == options.device(),                            \
            "structured kernels don't support multi-device outputs");       \
      } else {                                                              \
        guard_.reset_device(options.device());                              \
      }                                                                     \
      const auto& out = outputs_[output_idx].get();                         \
      check_inplace(out, sizes, options);                                   \
      if (!names.empty()) {                                                 \
        namedinference::propagate_names(outputs_[output_idx], names);       \
      }                                                                     \
      at::native::structured_bitwise_##name##_out ::set_output_raw_strided( \
          output_idx, sizes, strides, options, names);                      \
    }                                                                       \
    const Tensor& maybe_get_output(int64_t output_idx) override {           \
      return proxy_outputs_[output_idx].has_value()                         \
          ? **proxy_outputs_[output_idx]                                    \
          : outputs_[output_idx].get();                                     \
    }                                                                       \
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;                 \
    std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1>             \
        proxy_outputs_;                                                     \
    c10::musa::OptionalMUSAGuard guard_;                                    \
  };

#define DEFINE_STRUCTURED_BITWISE_OP(op_name)         \
  DEFINE_STRUCTURED_BITWISE_OP_MODE(op_name, out)     \
  DEFINE_STRUCTURED_BITWISE_OP_MODE(op_name, inplace) \
  DEFINE_STRUCTURED_BITWISE_OP_F(op_name)

DEFINE_STRUCTURED_BITWISE_OP(xor)
DEFINE_STRUCTURED_BITWISE_OP(and)
DEFINE_STRUCTURED_BITWISE_OP(or)

struct structured_xlogy_out_functional final
    : public at::native::structured_xlogy_out {
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_xlogy_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_xlogy_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }
  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_xlogy_out_inplace final
    : public at::native::structured_xlogy_out {
  structured_xlogy_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    check_inplace(out, sizes, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_xlogy_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    check_inplace(out, sizes, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_xlogy_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx].get();
  }

  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_xlogy_out_out final
    : public at::native::structured_xlogy_out {
  structured_xlogy_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_xlogy_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_xlogy_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx].get();
  }

  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};
} // namespace

at::Tensor xlogy_Tensor(const at::Tensor& self, const at::Tensor& other) {
  // No device check
  structured_xlogy_out_functional op;
  op.meta(self, other);
  op.impl(self, other, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

at::Tensor& xlogy__Tensor(at::Tensor& self, const at::Tensor& other) {
  // No device check
  structured_xlogy_out_inplace op(self);
  op.meta(self, other);
  op.impl(self, other, op.outputs_[0]);
  return self;
}

at::Tensor& xlogy_out_OutTensor(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  // No device check
  structured_xlogy_out_out op(out);
  op.meta(self, other);
  op.impl(self, other, op.maybe_get_output(0));
  return out;
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

struct structured_atan2_out_functional final
    : public at::native::structured_atan2_out {
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_atan2_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_atan2_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }
  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

at::Tensor Atan2(const at::Tensor& self, const at::Tensor& other) {
  // No device check
  structured_atan2_out_functional op;
  op.meta(self, other);
  op.impl(self, other, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

struct structured_atan2_out_out final
    : public at::native::structured_atan2_out {
  structured_atan2_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
    if (C10_UNLIKELY(maybe_proxy.has_value())) {
      proxy_outputs_[output_idx] =
          c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());
    }
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_atan2_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_atan2_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

at::Tensor& Atan2Out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  // No device check
  structured_atan2_out_out op(out);
  op.meta(self, other);
  op.impl(self, other, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}

struct structured_atan2_out_inplace final
    : public at::native::structured_atan2_out {
  structured_atan2_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    check_inplace(out, sizes, options);
    auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
    if (C10_UNLIKELY(maybe_proxy.has_value())) {
      proxy_outputs_[output_idx] =
          c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());
    }
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_atan2_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    check_inplace(out, sizes, options);
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_atan2_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

at::Tensor& Atan2_(at::Tensor& self, const at::Tensor& other) {
  // No device check
  structured_atan2_out_inplace op(self);
  op.meta(self, other);
  op.impl(self, other, op.outputs_[0]);
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return self;
}

ADVANCED_REGISTER(aten, PrivateUse1, "atan2", Atan2)
ADVANCED_REGISTER(aten, PrivateUse1, "atan2_", Atan2_)
ADVANCED_REGISTER(aten, PrivateUse1, "atan2.out", Atan2Out)

ADVANCED_REGISTER(aten, PrivateUse1, "add.Tensor", AddTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "add_.Tensor", Add_Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "add.out", Add_out)

ADVANCED_REGISTER(aten, PrivateUse1, "div.Tensor", DivTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "div.Tensor_mode", DivTensor_mode)
ADVANCED_REGISTER(aten, PrivateUse1, "div_.Tensor_mode", Div_Tensor_mode)
ADVANCED_REGISTER(aten, PrivateUse1, "div_.Tensor", Div_Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "div.out", Div_out)
ADVANCED_REGISTER(aten, PrivateUse1, "div.out_mode", Div_out_mode)

ADVANCED_REGISTER(aten, PrivateUse1, "eq.Tensor", EqualTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "eq_.Tensor", Equal_Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "eq.Tensor_out", Equal_out)

ADVANCED_REGISTER(aten, PrivateUse1, "equal", MUSAEqual)

ADVANCED_REGISTER(aten, PrivateUse1, "ge.Tensor", GreaterEqualTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "ge_.Tensor", GreaterEqual_Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "ge.Tensor_out", GreaterEqual_out)

ADVANCED_REGISTER(aten, PrivateUse1, "gt.Tensor", GreaterTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "gt_.Tensor", Greater_Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "gt.Tensor_out", Greater_out)

ADVANCED_REGISTER(aten, PrivateUse1, "mul.Tensor", MulTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "mul_.Tensor", Mul_Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "mul.out", Mul_out)

ADVANCED_REGISTER(aten, PrivateUse1, "minimum.out", Minimum_out)
ADVANCED_REGISTER(aten, PrivateUse1, "minimum.Tensor", MinimumTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "minimum_.Tensor", Minimum_Tensor)

ADVANCED_REGISTER(aten, PrivateUse1, "maximum.out", Maximum_out)
ADVANCED_REGISTER(aten, PrivateUse1, "maximum.Tensor", MaximumTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "maximum._Tensor", Maximum_Tensor)

ADVANCED_REGISTER(aten, PrivateUse1, "bitwise_xor.Tensor", Bitwise_xor_tensor)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "bitwise_xor_.Tensor",
    Bitwise_xor_tensor_inplace)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "bitwise_xor.Tensor_out",
    Bitwise_xor_tensor_out)

ADVANCED_REGISTER(aten, PrivateUse1, "bitwise_and.Tensor", Bitwise_and_tensor)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "bitwise_and_.Tensor",
    Bitwise_and_tensor_inplace)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "bitwise_and.Tensor_out",
    Bitwise_and_tensor_out)

ADVANCED_REGISTER(aten, PrivateUse1, "bitwise_or.Tensor", Bitwise_or_tensor)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "bitwise_or_.Tensor",
    Bitwise_or_tensor_inplace)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "bitwise_or.Tensor_out",
    Bitwise_or_tensor_out)

ADVANCED_REGISTER(aten, PrivateUse1, "ne.Tensor", NotEqualTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "ne_.Tensor", NotEqual_Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "ne.Tensor_out", NotEqual_out)
// not_equal, alias for torch.ne
REDEFINE_REGISTER(aten, PrivateUse1, "not_equal.Tensor", NotEqualTensor)
REDEFINE_REGISTER(aten, PrivateUse1, "not_equal_.Tensor", NotEqual_Tensor)
REDEFINE_REGISTER(aten, PrivateUse1, "not_equal.Tensor_out", NotEqual_out)

ADVANCED_REGISTER(aten, PrivateUse1, "sub.Tensor", SubTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "sub_.Tensor", Sub_Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "sub.out", Sub_out)

ADVANCED_REGISTER(aten, PrivateUse1, "remainder.Tensor", RemainderTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "remainder_.Tensor", Remainder_Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "remainder.Tensor_out", Remainder_out)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "remainder.Scalar_Tensor",
    RemainderScalarTensor)

ADVANCED_REGISTER(aten, PrivateUse1, "le.Tensor", LessEqualTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "le_.Tensor", LessEqual_Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "le.Tensor_out", LessEqual_out)

ADVANCED_REGISTER(aten, PrivateUse1, "lt.Tensor", LessTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "lt_.Tensor", Less_Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "lt.Tensor_out", Less_out)

REDEFINE_REGISTER(aten, PrivateUse1, "less.Tensor", LessTensor)
REDEFINE_REGISTER(aten, PrivateUse1, "less_.Tensor", Less_Tensor)
REDEFINE_REGISTER(aten, PrivateUse1, "less.Tensor_out", Less_out)

ADVANCED_REGISTER(aten, PrivateUse1, "silu_backward", SiluBwd)
ADVANCED_REGISTER(aten, PrivateUse1, "silu_backward.grad_input", SiluBwd_out)

ADVANCED_REGISTER(aten, PrivateUse1, "sigmoid_backward", SigmoidBwd)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "sigmoid_backward.grad_input",
    SigmoidBwd_out)

ADVANCED_REGISTER(aten, PrivateUse1, "tanh_backward", TanhBwd)
ADVANCED_REGISTER(aten, PrivateUse1, "tanh_backward.grad_input", TanhBwd_out)

ADVANCED_REGISTER(aten, PrivateUse1, "threshold_backward", ThresholdBwd)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "threshold_backward.grad_input",
    ThresholdBwd_out)

ADVANCED_REGISTER(aten, PrivateUse1, "gelu_backward", GELUBwd)
ADVANCED_REGISTER(aten, PrivateUse1, "gelu_backward.grad_input", GELUBwd_out)

ADVANCED_REGISTER(aten, PrivateUse1, "logical_and", LogicalAndTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "logical_and_", LogicalAnd_Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "logical_and.out", LogicalAnd_out)

ADVANCED_REGISTER(aten, PrivateUse1, "xlogy.Tensor", xlogy_Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "xlogy_.Tensor", xlogy__Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "xlogy.OutTensor", xlogy_out_OutTensor)

ADVANCED_REGISTER(aten, PrivateUse1, "pow.Tensor_Tensor_out", Pow_out)

ADVANCED_REGISTER(aten, PrivateUse1, "floor_divide", FloorDivideTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "floor_divide_.Tensor", FloorDivide_Tensor)
ADVANCED_REGISTER(aten, PrivateUse1, "floor_divide.out", FloorDivide_out)

} // namespace musa
} // namespace at
