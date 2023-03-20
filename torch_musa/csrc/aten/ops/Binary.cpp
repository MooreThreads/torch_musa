#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/BinaryOps.h>
#include <torch/library.h>
#pragma GCC diagnostic pop

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace native {
namespace {
using BINARY_MODE = ::musa::dnn::Binary::Mode;
using UNARY_MODE = ::musa::dnn::Unary::Mode;

void MusaBinaryCall(
    const std::string& op_name,
    Tensor& output,
    const Tensor& self,
    const Tensor& other,
    BINARY_MODE m = BINARY_MODE::ADD,
    const Scalar alpha_scalar = 1) {
  ::musa::dnn::Handle h;
  Tensor other_tmp = alpha_scalar.equal(1) ? other : at::empty_like(other);
  auto other_mt = CreateMUTensor(other_tmp);

  if (!alpha_scalar.equal(1)) {
    ::musa::dnn::Unary uop;
    CHECK_MUDNN_STATUS(uop.SetAlpha(alpha_scalar.toDouble()), "SetAlpha");
    CHECK_MUDNN_STATUS(uop.SetMode(UNARY_MODE::MUL), "SetMode");
    auto other_in = CreateMUTensor(other);
    CHECK_MUDNN_STATUS(uop.Run(h, other_mt, other_in), "Run " + op_name);
  }

  auto output_sizes = infer_size_dimvector(self.sizes(), other.sizes());
  output.resize_(output_sizes);

  ::musa::dnn::Binary bop;
  auto self_ = CreateMUTensor(self);
  auto om = CreateMUTensor(output);
  CHECK_MUDNN_STATUS(bop.SetMode(m), "SetMode");
  CHECK_MUDNN_STATUS(bop.Run(h, om, self_, other_mt), "Run " + op_name);
}

inline bool IsComparisonOp(const BINARY_MODE m) {
  return m == BINARY_MODE::EQ || m == BINARY_MODE::NE || m == BINARY_MODE::GE ||
      m == BINARY_MODE::GT || m == BINARY_MODE::LE || m == BINARY_MODE::LT;
}

Tensor MusaBinary(
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

  Tensor self_ = MusaContiguous(self);
  Tensor other_ = MusaContiguous(other);

  // In some special case that 'other' is scalar and on cpu, UnaryOp could take
  // the place of BinaryOp to optimize performance. such as:
  // torch.tensor([1.2, 2.4]) * 2.0
  auto SupportOptimizeScalarToUnary = [](BINARY_MODE m,
                                         const Tensor& input,
                                         const Tensor& other) {
    const bool support_mode_type =
        (m == BINARY_MODE::ADD && input.scalar_type() == ScalarType::Float) ||
        (m == BINARY_MODE::TRUEDIV &&
         input.scalar_type() == ScalarType::Float) ||
        (m == BINARY_MODE::MUL &&
         (input.scalar_type() == ScalarType::Float ||
          input.scalar_type() == ScalarType::Long));
    const bool support_dim_device =
        other.dim() == 0 && other.device() == DeviceType::CPU;
    return support_mode_type && support_dim_device;
  };
  const bool optimize_scalar_unary =
      SupportOptimizeScalarToUnary(m, self, other);
  if (!optimize_scalar_unary) {
    // 1. deal with other or self tensor on cpu
    // 2. deal with other and self tensor shape isn't same
    if (other.dim() == 0) {
      other_ = at::full_like(self_, other.item(), DeviceType::MTGPU);
    }
    if (self.dim() == 0) {
      self_ = at::full_like(other_, self.item(), DeviceType::MTGPU);
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
    if (self.dim() == 0) {
      output = MusaContiguous(self);
    } else {
      output = self_;
    }
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
    // Allocate output with bool dtype about some modes
    if (m == BINARY_MODE::EQ || m == BINARY_MODE::NE || m == BINARY_MODE::GE ||
        m == BINARY_MODE::GT || m == BINARY_MODE::LE || m == BINARY_MODE::LT ||
        m == BINARY_MODE::LOGICAL_AND || m == BINARY_MODE::LOGICAL_OR ||
        m == BINARY_MODE::LOGICAL_XOR) {
      output = empty_mtgpu(
          output_sizes,
          ScalarType::Bool,
          c10::nullopt,
          DeviceType::MTGPU,
          c10::nullopt,
          at::MemoryFormat::Contiguous);
    } else {
      output = empty_mtgpu(
          output_sizes,
          self_.scalar_type(),
          c10::nullopt,
          DeviceType::MTGPU,
          c10::nullopt,
          at::MemoryFormat::Contiguous);
    }
  }

  if (optimize_scalar_unary) {
    ::musa::dnn::Handle h;
    ::musa::dnn::Unary uop;
    auto other_scalar = other_.item();
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
    auto mt_output = CreateMUTensor(output);
    auto mt_input = CreateMUTensor(self_);
    CHECK_MUDNN_STATUS(uop.Run(h, mt_output, mt_input), "Run " + op_name);
  } else {
    MusaBinaryCall(op_name, output, self_, other_, m, alpha_scalar);
  }
  return inplace ? self.copy_(output) : output;
}

Tensor MusaBinarycommonDtype(
    const std::string& op_name,
    const Tensor& self,
    const Tensor& other,
    Scalar const& alpha_scalar,
    BINARY_MODE m) {
  auto commonDtype = at::result_type(self, other);
  alpha_check(commonDtype, alpha_scalar);
  Tensor self_ = self.to(commonDtype);
  Tensor other_ = other.to(commonDtype);
  return MusaBinary(op_name, self_, other_, m, alpha_scalar);
}

void MusaBinarycommonDtype_(
    const std::string& op_name,
    const Tensor& self,
    const Tensor& other,
    Scalar const& alpha_scalar,
    BINARY_MODE m) {
  auto commonDtype = at::result_type(self, other);
  alpha_check(commonDtype, alpha_scalar);
  Tensor other_ = other.to(commonDtype);
  MusaBinary(op_name, self, other_, m, alpha_scalar, true);
  return;
}

void MusaBinarycommonDtypeCall(
    const std::string& op_name,
    const Tensor& self,
    const Tensor& other,
    Scalar const& alpha_scalar,
    Tensor& output,
    BINARY_MODE m) {
  auto commonDtype = at::result_type(self, other);
  alpha_check(commonDtype, alpha_scalar);
  Tensor self_ = MusaContiguous(self.to(commonDtype));
  Tensor other_ = MusaContiguous(other.to(commonDtype));
  MusaBinaryCall(op_name, output, self_, other_, m, alpha_scalar);
}

Tensor MusaAddTensor(
    const Tensor& self,
    const Tensor& other,
    Scalar const& alpha_scalar) {
  return MusaBinarycommonDtype(
      __func__, self, other, alpha_scalar, BINARY_MODE::ADD);
}

Tensor& MusaAdd_Tensor(
    Tensor& self,
    const Tensor& other,
    Scalar const& alpha_scalar) {
  MusaBinarycommonDtype_(__func__, self, other, alpha_scalar, BINARY_MODE::ADD);
  return self;
}

Tensor& MusaAdd_out(
    const Tensor& self,
    const Tensor& other,
    Scalar const& alpha_scalar,
    Tensor& output) {
  MusaBinarycommonDtypeCall(
      __func__, self, other, alpha_scalar, output, BINARY_MODE::ADD);
  return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", &MusaAddTensor);
  m.impl("add_.Tensor", &MusaAdd_Tensor);
  m.impl("add.out", &MusaAdd_out);
}

} // namespace

} // namespace native
} // namespace at
