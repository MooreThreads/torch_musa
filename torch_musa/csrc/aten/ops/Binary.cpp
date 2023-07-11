#include <ATen/Config.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Activation.h>
#include <ATen/native/BinaryOps.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {
using BINARY_MODE = ::musa::dnn::Binary::Mode;
using UNARY_MODE = ::musa::dnn::Unary::Mode;

// TODO(MT AI) maybe we don't need this condition function when mudnn supports
bool NeedContiguous(const Tensor& self, const Tensor& other) {
  if (self.strides().equals(other.strides()) &&
      self.storage_offset() == other.storage_offset() &&
      self.sizes().equals(other.sizes())) {
    return false;
  }
  return true;
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
    const Scalar alpha_scalar = 1) {
  c10::musa::MUSAGuard device_guard(self.device());
  muHandle& h = GetMudnnHandle();
  Tensor other_tmp =
      alpha_scalar.equal(1) ? other : at::mul(other, alpha_scalar);
  auto output_sizes = infer_size_dimvector(self.sizes(), other.sizes());
  output.resize_(output_sizes);
  muTensor musa_self;
  muTensor musa_other;
  muTensor musa_out;
  Tensor contiguous_self;
  Tensor contiguous_other;
  Tensor contiguous_out;
  if (NeedContiguous(self, other)) {
    musa_self = CreateMUTensor(Contiguous(self, contiguous_self));
    musa_other = CreateMUTensor(Contiguous(other_tmp, contiguous_other));
    musa_out = CreateMUTensor(Contiguous(output, contiguous_out));
  } else {
    musa_self = CreateMUTensor(self, true);
    musa_other = CreateMUTensor(other_tmp, true);
    musa_out = CreateMUTensor(output, true);
  }

  ::musa::dnn::Binary bop;
  CHECK_MUDNN_STATUS(bop.SetMode(m), "SetMode");
  CHECK_MUDNN_STATUS(
      bop.Run(h, musa_out, musa_self, musa_other), "Run " + op_name);
}

inline bool IsComparisonOp(const BINARY_MODE m) {
  return m == BINARY_MODE::EQ || m == BINARY_MODE::NE || m == BINARY_MODE::GE ||
      m == BINARY_MODE::GT || m == BINARY_MODE::LE || m == BINARY_MODE::LT;
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
    Scalar const& alpha_scalar = 1,
    bool inplace = false) {
  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "input scalar type must the same");
  if (inplace) {
    TORCH_CHECK(
        is_expandable_to(other.sizes(), self.sizes()),
        "size {",
        self.sizes(),
        "} is not expandable to size {",
        other.sizes(),
        "}.");
  }

  if (self.numel() == 0 && other.numel() == 0) {
    if (IsComparisonOp(m)) {
      auto output_sizes = infer_size_dimvector(self.sizes(), other.sizes());
      return at::empty(output_sizes, self.options().dtype(ScalarType::Bool));
    } else {
      return at::empty({0}, self.options());
    }
  }

  // One of the both tensors might be cpu device. e.g.
  // 1. self will be in cpu if '1 + Tensor'.
  // 2. other wiil be in cpu if 'Tensor + 1'.
  // We use get musa devcie info to set context, so we need this check.
  Device device = is_musa(self) ? self.device() : other.device();
  c10::musa::MUSAGuard guard(device);

  // In some special case that 'other' is scalar and on cpu, UnaryOp could take
  // the place of BinaryOp to optimize performance. such as:
  // torch.tensor([1.2, 2.4]) * 2.0
  auto SupportOptimizeScalarToUnary =
      [](BINARY_MODE m, const Tensor& input, const Tensor& other) {
        const bool support_mode_type =
            (m == BINARY_MODE::ADD &&
             (input.scalar_type() == ScalarType::Float ||
              input.scalar_type() == ScalarType::Half)) ||
            (m == BINARY_MODE::TRUEDIV &&
             (input.scalar_type() == ScalarType::Float ||
              input.scalar_type() == ScalarType::Half)) ||
            (m == BINARY_MODE::MUL &&
             (input.scalar_type() == ScalarType::Float ||
              input.scalar_type() == ScalarType::Long ||
              input.scalar_type() == ScalarType::Half));
        const bool support_dim_device =
            other.dim() == 0 && other.device() == DeviceType::CPU;
        return support_mode_type && support_dim_device;
      };
  const bool optimize_scalar_unary =
      SupportOptimizeScalarToUnary(m, self, other);
  Tensor other_tmp = other;
  Tensor self_tmp = self;
  if (!optimize_scalar_unary) {
    // 1. deal with other or self tensor on cpu
    // 2. deal with other and self tensor shape isn't same
    if (other.dim() == 0) {
      other_tmp = at::full_like(self, other.item(), kMUSA);
    }
    if (self.dim() == 0) {
      self_tmp = at::full_like(other, self.item(), kMUSA);
    }
  }

  Tensor output;
  if (inplace) {
    TORCH_CHECK(
        is_expandable_to(other.sizes(), self.sizes()),
        "size {",
        self.sizes(),
        "} is not expandable to size {",
        other.sizes(),
        "}.");
    output = self_tmp;
  } else {
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
    if (m == BINARY_MODE::EQ || m == BINARY_MODE::NE || m == BINARY_MODE::GE ||
        m == BINARY_MODE::GT || m == BINARY_MODE::LE || m == BINARY_MODE::LT ||
        m == BINARY_MODE::LOGICAL_AND || m == BINARY_MODE::LOGICAL_OR ||
        m == BINARY_MODE::LOGICAL_XOR) {
      output_dtype = ScalarType::Bool;
    }
    if (NeedContiguous(self, other)) {
      output = at::empty(
          output_sizes,
          self.options()
              .dtype(output_dtype)
              .device(device)
              .memory_format(at::MemoryFormat::Contiguous));
    } else {
      output = empty_like(self, self.options().dtype(output_dtype));
    }
  }

  if (optimize_scalar_unary) {
    muHandle& h = GetMudnnHandle();
    ::musa::dnn::Unary uop;
    auto other_scalar = other_tmp.item();
    auto ConvertBinaryModeToString = [](BINARY_MODE mode) -> std::string {
      switch (mode) {
        case BINARY_MODE::ADD:
          return "Binary ADD";
        case BINARY_MODE::MUL:
          return "Binary MUL";
        case BINARY_MODE::TRUEDIV:
          return "Binary TRUEDIV";
        default:
          return "";
      }
    };
    if (other_scalar.isFloatingPoint()) {
      CHECK_MUDNN_STATUS(uop.SetAlpha(other_scalar.toDouble()), "SetAlpha");
    } else if (other_scalar.isIntegral(false)) {
      CHECK_MUDNN_STATUS(uop.SetAlpha(other_scalar.toLong()), "SetAlpha");
    } else {
      AT_ERROR(
          other_scalar.type(),
          " is not implemented for '",
          ConvertBinaryModeToString(m),
          "'!");
    }
    if (m == BINARY_MODE::MUL) {
      CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::MUL), "SetMode");
    } else if (m == BINARY_MODE::ADD) {
      CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::ADD), "SetMode");
    } else if (m == BINARY_MODE::TRUEDIV) {
      CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::DIV), "SetMode");
    }
    auto mt_output = CreateMUTensor(output, true);
    auto mt_input = CreateMUTensor(self_tmp, true);
    CHECK_MUDNN_STATUS(uop.Run(h, mt_output, mt_input), "Run " + op_name);
  } else {
    BinaryCall(op_name, output, self_tmp, other_tmp, m, alpha_scalar);
  }
  return inplace ? self.copy_(output) : output;
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
      return at::mul(self.cpu(), other.cpu()).to("musa");
    } else if (m == BINARY_MODE::TRUEDIV) {
      return at::div(self.cpu(), other.cpu()).to("musa");
    }
  }
  ScalarType common_dtype = at::result_type(self, other);
  at::native::alpha_check(common_dtype, alpha_scalar);
  Tensor contiguous_self = self.to(common_dtype);
  Tensor contiguous_other = other.to(common_dtype);
  return Binary(op_name, contiguous_self, contiguous_other, m, alpha_scalar);
}

void BinarycommonDtype_(
    const std::string& op_name,
    const Tensor& self,
    const Tensor& other,
    Scalar const& alpha_scalar,
    BINARY_MODE m) {
  ScalarType commonDtype = at::result_type(self, other);
  at::native::alpha_check(commonDtype, alpha_scalar);
  Tensor other_ = other.to(commonDtype);
  Binary(op_name, self, other_, m, alpha_scalar, true);
  return;
}

void BinarycommonDtypeInternal(
    const std::string& op_name,
    const Tensor& self,
    const Tensor& other,
    Scalar const& alpha_scalar,
    BINARY_MODE m) {
  ScalarType common_dtype = at::result_type(self, other);
  at::native::alpha_check(common_dtype, alpha_scalar);
  Tensor contiguous_other = other.to(common_dtype);
  Binary(op_name, self, contiguous_other, m, alpha_scalar, true);
  return;
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
  // WARN: output created by torch, which could be non-contiguous.
  output = at::musa::Contiguous(output);
  Tensor common_self = self.to(common_dtype);
  Tensor common_other = other.to(common_dtype);
  BinaryCall(op_name, output, common_self, common_other, m, alpha_scalar);
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
    BinarycommonDtypeInternal(__func__, self, other, alpha_scalar, mode);     \
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

// TODO(zaixing.wang): add check type, check device
#define DEFINE_BINARY_OP(op_name, mode)                             \
  Tensor op_name##Tensor(const Tensor& self, const Tensor& other) { \
    return BinarycommonDtype(__func__, self, other, 1, mode);       \
  }                                                                 \
                                                                    \
  Tensor& op_name##_Tensor(Tensor& self, const Tensor& other) {     \
    BinarycommonDtypeInternal(__func__, self, other, 1, mode);      \
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
    BinarycommonDtypeInternal(__func__, self, other, 1, BINARY_MODE::TRUEDIV);
  } else if (*rounding_mode == "trunc") {
    BinarycommonDtypeInternal(
        __func__, self, other, 1, BINARY_MODE::TRUNCATEDIV);
  } else if (*rounding_mode == "floor") {
    BinarycommonDtypeInternal(__func__, self, other, 1, BINARY_MODE::FLOORDIV);
  }
  return self;
}

Tensor RemainderScalarTensor(const Scalar& self, const Tensor& other) {
  Tensor other_cpu = other.cpu();
  Tensor out = at::remainder(self, other_cpu);
  auto out_musa = out.to("musa");
  return out_musa;
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
  auto contiguous_grad_output = Contiguous(grad_output);
  auto contiguous_self = Contiguous(self);

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
  auto result = ::at::empty(self.sizes(), self.options());
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
  c10::musa::MUSAGuard device_gaurd(self.device());
  auto contiguous_grad_output = Contiguous(grad_output);
  auto contiguous_self = Contiguous(self);

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
  auto grad_input = at::empty(self.sizes(), self.options());
  ThresholdBwd_out(grad_output, self, threshold, grad_input);
  return grad_input;
}

namespace {
struct structured_bitwise_and_out_out final
    : public at::native::structured_bitwise_and_out {
  structured_bitwise_and_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}

  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    at::native::structured_bitwise_and_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    at::native::structured_bitwise_and_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_bitwise_and_out_inplace final
    : public at::native::structured_bitwise_and_out {
  structured_bitwise_and_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
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
    at::native::structured_bitwise_and_out::set_output_raw_strided(
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
    at::native::structured_bitwise_and_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_bitwise_and_out_functional final
    : public at::native::structured_bitwise_and_out {
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
    at::native::structured_bitwise_and_out::set_output_raw_strided(
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
    at::native::structured_bitwise_and_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }
  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

// TODO: porting code, when codegens, its will do the refactor later.
struct structured_bitwise_or_out_out final
    : public at::native::structured_bitwise_or_out {
  structured_bitwise_or_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}

  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    at::native::structured_bitwise_or_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    at::native::structured_bitwise_or_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_bitwise_or_out_inplace final
    : public at::native::structured_bitwise_or_out {
  structured_bitwise_or_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}
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
    at::native::structured_bitwise_or_out::set_output_raw_strided(
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
    at::native::structured_bitwise_or_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_bitwise_or_out_functional final
    : public at::native::structured_bitwise_or_out {
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
    at::native::structured_bitwise_or_out::set_output_raw_strided(
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
    at::native::structured_bitwise_or_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }
  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

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

struct structured_hardsigmoid_backward_out_functional final
    : public at::native::structured_hardsigmoid_backward_out {
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
    at::native::structured_hardsigmoid_backward_out::set_output_raw_strided(
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
    at::native::structured_hardsigmoid_backward_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }
  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_hardsigmoid_backward_out_out final
    : public at::native::structured_hardsigmoid_backward_out {
  structured_hardsigmoid_backward_out_out(Tensor& out0)
      : outputs_{std::ref(out0)} {}
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
    at::native::structured_hardsigmoid_backward_out::set_output_raw_strided(
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
    at::native::structured_hardsigmoid_backward_out::set_output_raw_strided(
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
} // namespace

Tensor& BitWise_AndTensor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  // No device check

  structured_bitwise_and_out_out op(out);
  op.meta(self, other);
  op.impl(self, other, op.maybe_get_output(0));
  return out;
}

at::Tensor& BitWise_And_Tensor(at::Tensor& self, const at::Tensor& other) {
  // No device check
  structured_bitwise_and_out_inplace op(self);
  op.meta(self, other);
  op.impl(self, other, op.outputs_[0]);
  return self;
}

at::Tensor BitWise_AndTensor(const at::Tensor& self, const at::Tensor& other) {
  // No device check
  structured_bitwise_and_out_functional op;
  op.meta(self, other);
  op.impl(self, other, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

Tensor& BitWise_OrTensor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  // No device check

  structured_bitwise_or_out_out op(out);
  op.meta(self, other);
  op.impl(self, other, op.maybe_get_output(0));
  return out;
}

at::Tensor& BitWise_Or_Tensor(at::Tensor& self, const at::Tensor& other) {
  // No device check
  structured_bitwise_or_out_inplace op(self);
  op.meta(self, other);
  op.impl(self, other, op.outputs_[0]);
  return self;
}

at::Tensor BitWise_OrTensor(const at::Tensor& self, const at::Tensor& other) {
  // No device check
  structured_bitwise_or_out_functional op;
  op.meta(self, other);
  op.impl(self, other, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

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

at::Tensor HardSigmoidBwd(
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  structured_hardsigmoid_backward_out_functional op;
  op.meta(grad_output, self);
  op.impl(grad_output, self, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

at::Tensor& HardSigmoidBwd_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Tensor& grad_input) {
  structured_hardsigmoid_backward_out_out op(grad_input);
  op.meta(grad_output, self);
  op.impl(grad_output, self, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return grad_input;
}

Tensor HardSwishBwd(const Tensor& grad_output, const Tensor& self) {
  return at::native::hardswish_backward(grad_output, self);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", &AddTensor);
  m.impl("add_.Tensor", &Add_Tensor);
  m.impl("add.out", &Add_out);

  m.impl("div.Tensor", &DivTensor);
  m.impl("div.Tensor_mode", &DivTensor_mode);
  m.impl("div_.Tensor_mode", &Div_Tensor_mode);
  m.impl("div_.Tensor", &Div_Tensor);
  m.impl("div.out", &Div_out);
  m.impl("div.out_mode", &Div_out_mode);

  m.impl("eq.Tensor", &EqualTensor);
  m.impl("eq_.Tensor", &Equal_Tensor);
  m.impl("eq.Tensor_out", &Equal_out);

  m.impl("ge.Tensor", &GreaterEqualTensor);
  m.impl("ge_.Tensor", &GreaterEqual_Tensor);
  m.impl("ge.Tensor_out", &GreaterEqual_out);

  m.impl("gt.Tensor", &GreaterTensor);
  m.impl("gt_.Tensor", &Greater_Tensor);
  m.impl("gt.Tensor_out", &Greater_out);

  m.impl("mul.Tensor", &MulTensor);
  m.impl("mul_.Tensor", &Mul_Tensor);
  m.impl("mul.out", &Mul_out);

  m.impl("minimum.out", &Minimum_out);
  m.impl("minimum.Tensor", &MinimumTensor);
  m.impl("minimum_.Tensor", &Minimum_Tensor);

  m.impl("maximum.out", &Maximum_out);
  m.impl("maximum.Tensor", &MaximumTensor);
  m.impl("maximum._Tensor", &Maximum_Tensor);

  m.impl("bitwise_and.Tensor", &BitWise_AndTensor);
  m.impl("bitwise_and_.Tensor", &BitWise_And_Tensor);
  m.impl("bitwise_and.Tensor_out", &BitWise_AndTensor_out);

  m.impl("bitwise_or.Tensor", &BitWise_OrTensor);
  m.impl("bitwise_or_.Tensor", &BitWise_Or_Tensor);
  m.impl("bitwise_or.Tensor_out", &BitWise_OrTensor_out);

  m.impl("ne.Tensor", &NotEqualTensor);
  m.impl("ne_.Tensor", &NotEqual_Tensor);
  m.impl("ne.Tensor_out", &NotEqual_out);
  // not_equal, alias for torch.ne
  m.impl("not_equal.Tensor", &NotEqualTensor);
  m.impl("not_equal_.Tensor", &NotEqual_Tensor);
  m.impl("not_equal.Tensor_out", &NotEqual_out);

  m.impl("sub.Tensor", &SubTensor);
  m.impl("sub_.Tensor", &Sub_Tensor);
  m.impl("sub.out", &Sub_out);

  m.impl("remainder.Tensor", &RemainderTensor);
  m.impl("remainder_.Tensor", &Remainder_Tensor);
  m.impl("remainder.Tensor_out", &Remainder_out);
  m.impl("remainder.Scalar_Tensor", &RemainderScalarTensor);

  m.impl("le.Tensor", &LessEqualTensor);
  m.impl("le_.Tensor", &LessEqual_Tensor);
  m.impl("le.Tensor_out", &LessEqual_out);

  m.impl("lt.Tensor", &LessTensor);
  m.impl("lt_.Tensor", &Less_Tensor);
  m.impl("lt.Tensor_out", &Less_out);
  m.impl("less.Tensor", &LessTensor);
  m.impl("less_.Tensor", &Less_Tensor);
  m.impl("less.Tensor_out", &Less_out);
  m.impl("silu_backward", &SiluBwd);
  m.impl("silu_backward.grad_input", &SiluBwd_out);

  m.impl("sigmoid_backward", &SigmoidBwd);
  m.impl("sigmoid_backward.grad_input", &SigmoidBwd_out);

  m.impl("tanh_backward", &TanhBwd);
  m.impl("tanh_backward.grad_input", &TanhBwd_out);

  m.impl("threshold_backward", &ThresholdBwd);
  m.impl("threshold_backward.grad_input", &ThresholdBwd_out);

  m.impl("gelu_backward", &GELUBwd);
  m.impl("gelu_backward.grad_input", &GELUBwd_out);

  m.impl("logical_and", &LogicalAndTensor);
  m.impl("logical_and_", &LogicalAnd_Tensor);
  m.impl("logical_and.out", &LogicalAnd_out);

  m.impl("xlogy.Tensor", &xlogy_Tensor);
  m.impl("xlogy_.Tensor", &xlogy__Tensor);
  m.impl("xlogy.OutTensor", &xlogy_out_OutTensor);

  m.impl("pow.Tensor_Tensor_out", &Pow_out);

  m.impl("hardswish_backward", &HardSwishBwd);

  m.impl("hardsigmoid_backward", &HardSigmoidBwd);
  m.impl("hardsigmoid_backward.grad_input", &HardSigmoidBwd_out);

  m.impl("floor_divide", &FloorDivideTensor);
  m.impl("floor_divide_.Tensor", &FloorDivide_Tensor);
  m.impl("floor_divide.out", &FloorDivide_out);
}

} // namespace musa
} // namespace at
