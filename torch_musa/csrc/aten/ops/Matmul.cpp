#include <ATen/Config.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation_native.h>
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/vdot_native.h>
#include <ATen/ops/view_as_complex.h>
#include <ATen/ops/zeros.h>
#include "ATen/ops/gelu_native.h"
#include "ATen/ops/relu_native.h"
#endif

#include <mudnn.h>

#include "ATen/core/TensorBody.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Exception.h"

#include "torch_musa/csrc/aten/musa/MUSABlas.h"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

namespace {
bool is_broadcasted_except_last_dim(const at::Tensor& t) {
  if (t.dim() == 0 || t.dim() == 1) {
    return false;
  }
  bool choose = (t.stride(-1) == 1);
  for (int i = 0; i < t.dim() - 1; i++) {
    choose &= (t.stride(i) == 0);
    if (!choose) {
      break;
    }
  }
  return choose;
}
} // namespace

at::Tensor& DotOut(const at::Tensor& l, const at::Tensor& r, at::Tensor& out) {
  TORCH_CHECK(l.sizes() == r.sizes(), "dot tensors' shape don't match");
  TORCH_CHECK(
      l.dim() == r.dim() && l.dim() == 1, "dot inputs must be 1-D tensors");
  const c10::musa::MUSAGuard device_guard(l.device());

  muHandle& h = GetMudnnHandle();
  if (l.numel() == 0 || r.numel() == 0) {
    out.zero_().squeeze_();
    return out;
  }
  auto rst = CreateMUTensor(out);
  Tensor contiguous_l = l.contiguous();
  Tensor contiguous_r = r.contiguous();
  auto lmt = CreateMUTensor(contiguous_l);
  auto rmt = CreateMUTensor(contiguous_r);

  ::musa::dnn::Dot op;
  op.SetComputeMode(at::musa::GetComputeModeFromCtx(l.scalar_type()));
  CHECK_MUDNN_STATUS(op.Run(h, rst, lmt, rmt, InternalMemAlloc), "Run")

  out.squeeze_();
  return out;
}

at::Tensor Dot(const at::Tensor& l, const at::Tensor& r) {
  Tensor out =
      at::empty({1}, l.options().memory_format(at::MemoryFormat::Contiguous));

  DotOut(l, r, out);
  return out;
}

namespace {

void DotCheck(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      self.dim() == 1 && other.dim() == 1,
      "1D tensors expected, but got ",
      self.dim(),
      "D and ",
      other.dim(),
      "D tensors");
  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "dot : expected both vectors to have same dtype, but found ",
      self.scalar_type(),
      " and ",
      other.scalar_type());
  TORCH_CHECK(
      self.numel() == other.numel(),
      "inconsistent tensor size, expected tensor [",
      self.numel(),
      "] and src [",
      other.numel(),
      "] to have the same number of elements, but got ",
      self.numel(),
      " and ",
      other.numel(),
      " elements respectively");
  TORCH_CHECK(
      (self.numel() <= INT_MAX) && (self.stride(0) <= INT_MAX) &&
          (other.stride(0) <= INT_MAX),
      "dot only supports n, incx, incy with the bound [val] <= %d",
      INT_MAX);
}

} // anonymous namespace

at::Tensor VDot(const Tensor& l, const Tensor& r) {
  DotCheck(l, r);

  if (!l.is_complex()) {
    return Dot(l, r);
  }

  if (l._is_zerotensor() || r._is_zerotensor()) {
    return at::_efficientzerotensor({}, l.options());
  }

  at::ScalarType real_dtype = c10::toRealValueType(l.scalar_type());

  if (l.numel() == 0) {
    Tensor x_real = at::zeros({2}, l.options().dtype(real_dtype));
    return at::view_as_complex(x_real);
  }

  const c10::musa::MUSAGuard device_guard(l.device());

  const int n = static_cast<int>(l.numel());
  int incx = static_cast<int>(l.stride(0));
  int incy = static_cast<int>(r.stride(0));
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  return AT_DISPATCH_COMPLEX_TYPES(l.scalar_type(), "vdot", [&] {
    Tensor result = at::empty({}, l.options());

    at::musa::blas::vdot<scalar_t>(
        n,
        l.const_data_ptr<scalar_t>(),
        incx,
        r.const_data_ptr<scalar_t>(),
        incy,
        result.mutable_data_ptr<scalar_t>());

    return result;
  });
}
// TODO: support vdot.out ops

void BlasGEMM(
    const Tensor& l,
    const Tensor& r,
    const c10::optional<Tensor>& bias,
    Tensor& out,
    const Scalar& alpha = 1,
    const Scalar& beta = 0) {
  // blas use column major
  TORCH_CHECK(l.is_contiguous(), "left hand tensor needs to be contiguous");
  TORCH_CHECK(r.is_contiguous(), "right hand tensor needs to be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out tensor needs to be contiguous");

  // (m, k) x (k, n) ---> (m, n)
  // (n, k) x (k, m) ---> (n, m)
  int64_t m = out.size(0);
  int64_t n = out.size(1);
  int64_t k = l.size(1);

  AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "musaBlasGEMM", [&] {
    const auto alpha_value = alpha.to<scalar_t>();
    const auto beta_value = beta.to<scalar_t>();
    at::musa::blas::gemm(
        'n',
        'n',
        n,
        m,
        k,
        alpha_value,
        r.const_data_ptr<scalar_t>(),
        n,
        l.const_data_ptr<scalar_t>(),
        k,
        beta_value,
        out.data_ptr<scalar_t>(),
        n);
  });
}

void MmCall(
    const Tensor& l,
    const Tensor& r,
    const std::optional<Tensor>& bias,
    Tensor& out,
    const at::Scalar& alpha = 1,
    const at::Scalar beta = 0) {
  if C10_UNLIKELY (l.numel() == 0 || r.numel() == 0) {
    if (!bias.has_value()) {
      out.zero_();
    } else if (bias.value().is_same(out)) {
      out.mul_(beta);
    } else {
      out.zero_();
      out.add_(bias.value(), beta);
    }
    return;
  }

  muHandle& h = GetMudnnHandle();
  bool trans_l = IsTranspose(l);
  bool trans_r = IsTranspose(r);

  // if IsTranspose(mat) is True, we don't need to clone to permutate memory
  Tensor contiguous_l;
  Tensor contiguous_r;

  // muDNN need origin mat shape info, so we need to transpose(-2, -1) here
  auto lmt = trans_l ? CreateMUTensor(l.transpose(-2, -1))
                     : CreateMUTensor(ContiguousRef(l, contiguous_l));
  auto rmt = trans_r ? CreateMUTensor(r.transpose(-2, -1))
                     : CreateMUTensor(ContiguousRef(r, contiguous_r));
  auto out_contig = out.contiguous();
  auto rst = CreateMUTensor(out_contig);

  ::musa::dnn::MatMul mm;
  CHECK_MUDNN_STATUS(
      mm.SetComputeMode(at::musa::GetComputeModeFromCtx(l.scalar_type())),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(mm.SetTranspose(trans_l, trans_r), "SetTranspose");

  // For bias that have shape (s0, s1, s2, ... , sn) and stride (0, 0, 0, ...,
  // 1), we can delegate this case into second if block to reduce memory
  // allocation
  bool bias_broadcasted_except_last_dim =
      bias.has_value() ? is_broadcasted_except_last_dim(bias.value()) : false;
  if (bias.has_value() && bias->sizes() == out.sizes() &&
      !bias_broadcasted_except_last_dim) {
    // For both inplace and outplace, we run muDNN MM with `d = alpha * a @ b +
    // beta * c + gamma * bias`, of which the bias is omitted
    const auto bias_ =
        FormatContiguous(bias.value(), out.suggest_memory_format());
    auto bmt = CreateMUTensor(bias_);
    CHECK_MUDNN_STATUS(mm.SetAlpha(alpha.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(mm.SetBeta(beta.to<double>()), "SetBeta");
    CHECK_MUDNN_STATUS(
        mm.RunWithBiasAdd(h, rst, lmt, rmt, bmt, muTensor(), InternalMemAlloc),
        "RunWithBiasAdd");
  } else if (
      bias.has_value() &&
      (bias->dim() == 1 || bias_broadcasted_except_last_dim)) {
    // TODO(@mt-ai): should we check the bias is broadcastable?
    // Run muDNN MM with `d = alpha * a @ b + beta * c + gamma * bias`, of
    // which c == d
    const auto bias_t = bias.value();
    const auto bias_t_ =
        bias_t.as_strided({bias_t.size(-1)}, {bias_t.stride(-1)});
    auto bmt = CreateMUTensor(bias_t_);
    CHECK_MUDNN_STATUS(mm.SetAlpha(alpha.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(mm.SetGamma(beta.to<double>()), "SetGamma");
    CHECK_MUDNN_STATUS(
        mm.RunWithBiasAdd(h, rst, lmt, rmt, rst, bmt, InternalMemAlloc),
        "RunWithBiasAdd");
  } else {
    // Run muDNN with `c = alpha * a @ b + beta * c`, then `c += gamma * bias`
    // if bias is given (scalar or [M, 1] for gemm)
    CHECK_MUDNN_STATUS(mm.SetAlpha(alpha.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(mm.Run(h, rst, lmt, rmt, InternalMemAlloc), "Run");
    if (bias.has_value()) {
      out.add_(bias.value(), beta);
    }
  }

  if (!out.is_contiguous()) {
    out.copy_(out_contig);
  }
}

void BmmCall(
    const Tensor& l,
    const Tensor& r,
    Tensor& out,
    const at::Scalar& alpha = 1,
    const at::Scalar beta = 0) {
  if C10_UNLIKELY (l.numel() == 0 || r.numel() == 0) {
    out.zero_();
    return;
  }

  muHandle& h = GetMudnnHandle();
  bool trans_l = IsTranspose(l);
  bool trans_r = IsTranspose(r);

  // if IsTranspose(mat) is True, we don't need to clone to permutate memory
  Tensor contiguous_l;
  Tensor contiguous_r;

  // muDNN need origin mat shape info, so we need to transpose(-2, -1) here
  auto lmt = trans_l ? CreateMUTensor(l.transpose(-2, -1))
                     : CreateMUTensor(ContiguousRef(l, contiguous_l));
  auto rmt = trans_r ? CreateMUTensor(r.transpose(-2, -1))
                     : CreateMUTensor(ContiguousRef(r, contiguous_r));
  auto rst = CreateMUTensor(out);

  // Run muDNN BMM with `c = alpha * a @ b + beta * c`
  ::musa::dnn::BatchMatMul bmm;
  CHECK_MUDNN_STATUS(
      bmm.SetComputeMode(at::musa::GetComputeModeFromCtx(l.scalar_type())),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(bmm.SetTranspose(trans_l, trans_r), "SetTranspose");
  CHECK_MUDNN_STATUS(bmm.SetAlpha(alpha.to<double>()), "SetAlpha");
  CHECK_MUDNN_STATUS(bmm.SetBeta(beta.to<double>()), "SetBeta");
  CHECK_MUDNN_STATUS(bmm.Run(h, rst, lmt, rmt, InternalMemAlloc), "Run");
}

at::Tensor& AddMmOut(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  const auto device_guard = c10::musa::MUSAGuard(self.device());
  TORCH_CHECK(
      mat1.dim() == 2 && mat2.dim() == 2 && mat1.size(1) == mat2.size(0),
      "mat1 and mat2 must be a matrix and mat1_shape[1](",
      mat1.size(1),
      ") must equal to "
      "mat2_shape[0](",
      mat2.size(0),
      ")");
  TORCH_CHECK(
      self.dim() != 1 || self.size(0) == out.size(1),
      "bias with dim=1 should match out_shape[1]");
  MmCall(mat1, mat2, self, out, alpha, beta);
  return out;
}

at::Tensor& AddMm_(
    at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  AddMmOut(self, mat1, mat2, beta, alpha, self);
  return self;
}

at::Tensor AddMm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  Tensor result = at::empty(
      {mat1.size(0), mat2.size(1)},
      self.options().memory_format(at::MemoryFormat::Contiguous));
  AddMmOut(self, mat1, mat2, beta, alpha, result);
  return result;
}

at::Tensor& AddMvOut(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  TORCH_CHECK(
      mat.dim() == 2 && vec.dim() == 1 && mat.size(1) == vec.size(0),
      "mat and vec must be a matrix and mat1_shape[1] must equal to "
      "vec[0]");
  TORCH_CHECK(
      out.dim() == 1 && out.size(0) == mat.size(0),
      "out shape doesn't match mat[0]");
  TORCH_CHECK(
      self.dim() != 1 || self.size(0) == 1 || self.size(0) == mat.size(0),
      "addmv bias with dim=1 should have size of [1] of mat_size[0]");
  if (self.dim() == 0 && self.numel() == 1) {
    MmCall(mat, vec, self.view({-1}), out, alpha, beta);
  } else {
    MmCall(mat, vec, self, out, alpha, beta);
  }
  return out;
}

at::Tensor& AddMv_(
    at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  AddMvOut(self, mat, vec, beta, alpha, self);
  return self;
}

at::Tensor AddMv(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  at::Tensor result = at::empty({mat.size(0)}, mat.options());
  AddMvOut(self, mat, vec, beta, alpha, result);
  return result;
}

Tensor& MmOut(const Tensor& self, const Tensor& mat2, Tensor& out) {
  const auto device_guard = c10::musa::MUSAGuard(self.device());
  TORCH_CHECK(
      self.dim() == 2 && mat2.dim() == 2 && self.size(1) == mat2.size(0),
      "self and mat2 must be a matrix and self_shape[1] must equal to "
      "mat2_shape[0]");
  if (out.scalar_type() == ScalarType::Double) {
    BlasGEMM(self, mat2, c10::nullopt, out);
  } else {
    MmCall(self, mat2, c10::nullopt, out);
  }
  return out;
}

Tensor Mm(const Tensor& self, const Tensor& mat2) {
  Tensor result = at::empty(
      {self.size(0), mat2.size(1)},
      self.options().memory_format(at::MemoryFormat::Contiguous));
  MmOut(self, mat2, result);
  return result;
}

Tensor& MvOut(const Tensor& self, const Tensor& vec, Tensor& out) {
  const auto device_guard = c10::musa::MUSAGuard(self.device());
  TORCH_CHECK(
      self.dim() == 2 && vec.dim() == 1 && self.size(1) == vec.size(0),
      "self and vec must be a matrix and a vector, and self_shape[1] must equal to "
      "mat2_shape[0]");
  MmCall(self, vec, c10::nullopt, out);
  return out;
}

Tensor Mv(const Tensor& self, const Tensor& vec) {
  Tensor result = at::empty(
      {self.size(0)},
      self.options().memory_format(at::MemoryFormat::Contiguous));
  MvOut(self, vec, result);
  return result;
}

Tensor& BmmOut(const Tensor& self, const Tensor& mat2, Tensor& out) {
  const auto device_guard = c10::musa::MUSAGuard(self.device());
  TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3, "self must be a 3D matrix");
  TORCH_CHECK(
      self.size(0) == mat2.size(0) && self.size(2) == mat2.size(1),
      "self_shape[0] must equal to mat2_shape[0], and self_shape[2] "
      "must equal to mat2_shape[1]");
  BmmCall(self, mat2, out);
  return out;
}

Tensor Bmm(const Tensor& self, const Tensor& mat2) {
  Tensor result = at::empty(
      {self.size(0), self.size(1), mat2.size(2)},
      self.options().memory_format(at::MemoryFormat::Contiguous));
  BmmOut(self, mat2, result);
  return result;
}

// Computes matrix multiply + bias while applying scaling to input and output
// matrices and computes amax Scales are only applicable when matrices are of
// Float8 type and assumbed to be equal to 1.0 by default. If output matrix type
// is 16 or 32-bit type, neither scale_result is applied nor amax is computed.
// Known limitations:
//  - Only works if mat1 is row-major and mat2 is column-major
//  - Only works if matrices sizes are divisible by 32
Tensor& ScaledMatmulOut(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& scale_a,
    const Tensor& scale_b,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& scale_result,
    std::optional<c10::ScalarType> out_dtype,
    bool use_fast_accum,
    Tensor& out) {
  const c10::musa::MUSAGuard device_guard(mat1.device());
  if (at::musa::getMUSAArch() < 310) {
    TORCH_CHECK(false, "scaled_gemm is only supported for MUSA_ARCH >= 310");
  }

  // Check sizes
  auto dprops = at::musa::getCurrentDeviceProperties();
  TORCH_CHECK(
      dprops->major >= 3,
      "torch._scaled_mm on MUSA is only supported on devices with compute capability >= 3.1)");
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      " and ",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ")");
  TORCH_CHECK(
      (scale_a.numel() == 1 || scale_a.numel() == mat1.size(0)) &&
          scale_a.scalar_type() == kFloat,
      "scale_a must be float scalar or float tensor of shape ",
      mat1.size(0),
      " but got ",
      scale_a.numel());
  TORCH_CHECK(
      (scale_b.numel() == 1 || scale_b.numel() == mat2.size(1)) &&
          scale_b.scalar_type() == kFloat,
      "scale_b must be float scalar or float tensor of shape ",
      mat2.size(1),
      " but got ",
      scale_b.numel());
  TORCH_CHECK(
      !scale_result ||
          ((scale_result->numel() == 1 ||
            scale_result->numel() == mat1.size(0)) &&
           scale_result->scalar_type() == kFloat),
      "scale_result must be float scalar or float tensor of shape ",
      mat1.size(0),
      " but got ",
      scale_result->numel());
  TORCH_CHECK(
      !bias || bias->numel() == mat2.size(1),
      "Bias must be size ",
      mat2.size(1),
      " but got ",
      bias->numel());
  // Check types
  TORCH_CHECK(
      !out_dtype || *out_dtype == out.scalar_type(),
      "out_dtype must match output matrix type");
  TORCH_CHECK(
      c10::isFloat8Type(mat1.scalar_type()),
      "Expected mat1 to be Float8 matrix got ",
      mat1.scalar_type());
  TORCH_CHECK(
      c10::isFloat8Type(mat2.scalar_type()),
      "Expected mat2 to be Float8 matrix got ",
      mat2.scalar_type());
  if (bias) {
    TORCH_CHECK(
        (out.scalar_type() == ScalarType::Float &&
         bias->scalar_type() == ScalarType::Float) ||
            (out.scalar_type() != ScalarType::Float &&
             (bias->scalar_type() == ScalarType::BFloat16 ||
              bias->scalar_type() == ScalarType::Half)),
        "Wrong bias dtype: ",
        bias->scalar_type(),
        "which doesn't match out dtype: ",
        out.scalar_type());
  }
  Tensor bias_ = bias.value_or(Tensor());
  Tensor scale_result_ = scale_result.value_or(Tensor());
  TensorArg targs[]{
      {out, "out", 0},
      {mat1, "mat1", 1},
      {mat2, "mat2", 2},
      {bias_, "bias", 3},
      {scale_a, "scale_a", 4},
      {scale_b, "scale_b", 5},
      {scale_result_, "scale_result", 6}};
  checkAllSameGPU(__func__, targs);

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  at::native::resize_output(out, {mat1_sizes[0], mat2_sizes[1]});

  muHandle& h = GetMudnnHandle();
  Tensor contiguous_l;
  Tensor contiguous_r;
  bool trans_l = IsTranspose(mat1);
  bool trans_r = IsTranspose(mat2);
  muTensor lmt = trans_l ? CreateMUTensor(mat1.transpose(-2, -1))
                         : CreateMUTensor(ContiguousRef(mat1, contiguous_l));
  muTensor rmt = trans_r ? CreateMUTensor(mat2.transpose(-2, -1))
                         : CreateMUTensor(ContiguousRef(mat2, contiguous_r));
  muTensor bmt = CreateMUTensor(bias_);
  muTensor rst = CreateMUTensor(out);
  muTensor sa = CreateMUTensor(scale_a);
  muTensor sb = CreateMUTensor(scale_b);
  muTensor sr = CreateMUTensor(scale_result_);

  Tensor amax = at::empty({0}, mat1.options().dtype(ScalarType::Float));
  // if we don't need amax, just replace below with:
  // muTensor amax_ = muTensor();
  muTensor amax_ = CreateMUTensor(amax);

  ::musa::dnn::BatchMatMul op;
  CHECK_MUDNN_STATUS(
      op.SetComputeMode(at::musa::GetComputeModeFromCtx(mat1.scalar_type())),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(op.SetTranspose(trans_l, trans_r), "SetTranspose");

  ::musa::dnn::MatMulLtParam param;
  CHECK_MUDNN_STATUS(param.SetScale(sa, sb, muTensor(), sr), "SetScale");
  CHECK_MUDNN_STATUS(param.SetAmaxD(amax_), "SetAmax");

  CHECK_MUDNN_STATUS(
      op.RunLt(h, rst, lmt, rmt, rst, bmt, param, InternalMemAlloc), "RunLt");

  return out;
}

Tensor ScaledMatmul(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& scale_a,
    const Tensor& scale_b,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& scale_result,
    std::optional<c10::ScalarType> out_dtype,
    bool use_fast_accum) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  Tensor out = at::empty({0}, mat_a.options().dtype(out_dtype_));
  return ScaledMatmulOut(
      mat_a,
      mat_b,
      scale_a,
      scale_b,
      bias,
      scale_result,
      out_dtype,
      use_fast_accum,
      out);
}

TORCH_IMPL_FUNC(addmm_activation_out_musa)
(const Tensor& self,
 const Tensor& mat1,
 const Tensor& mat2,
 const Scalar& beta,
 const Scalar& alpha,
 bool use_gelu,
 const Tensor& result) {
  // self.size: (k,) || (m, n)
  // mat1.size: (m, k)
  // mat2.size: (k, n)
  // result.size: (m, n)
  AddMmOut(self, mat1, mat2, beta, alpha, const_cast<at::Tensor&>(result));
  if (use_gelu) {
    // avoid dispatch here
    at::musa::Gelu_(const_cast<at::Tensor&>(result), "tanh");
  } else {
    at::musa::Relu_(const_cast<at::Tensor&>(result));
  }
}

} // namespace musa
} // namespace at
