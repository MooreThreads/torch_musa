#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include <mudnn.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

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

void MmCall(
    const Tensor& l,
    const Tensor& r,
    const c10::optional<Tensor>& bias,
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
  auto rst = CreateMUTensor(out);

  ::musa::dnn::MatMul mm;
  CHECK_MUDNN_STATUS(
      mm.SetComputeMode(at::musa::GetComputeModeFromCtx(l.scalar_type())),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(mm.SetTranspose(trans_l, trans_r), "SetTranspose");

  if (bias.has_value() && bias->sizes() == out.sizes()) {
    // For both inplace and outplace, we run muDNN MM with `d = alpha * a @ b +
    // beta * c + gamma * bias`, of which the bias is omitted
    auto bmt = CreateMUTensor(bias.value());
    CHECK_MUDNN_STATUS(mm.SetAlpha(alpha.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(mm.SetBeta(beta.to<double>()), "SetBeta");
    CHECK_MUDNN_STATUS(
        mm.RunWithBiasAdd(h, rst, lmt, rmt, bmt, muTensor(), InternalMemAlloc),
        "RunWithBiasAdd");
  } else if (bias.has_value() && bias->dim() == 1) {
    // TODO(@mt-ai): should we check the bias is broadcastable?
    // Run muDNN MM with `d = alpha * a @ b + beta * c + gamma * bias`, of
    // which c == d
    auto bmt = CreateMUTensor(bias.value());
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
  MmCall(self, mat2, c10::nullopt, out);
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

} // namespace musa
} // namespace at
