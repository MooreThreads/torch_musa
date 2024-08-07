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
    bool is_batch = false,
    const at::Scalar& alpha = 1,
    const at::Scalar beta = 0,
    const at::Scalar gama = 0) {
  if (l.numel() == 0 || r.numel() == 0) {
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

  if (is_batch) {
    ::musa::dnn::BatchMatMul b_mm;
    CHECK_MUDNN_STATUS(
        b_mm.SetComputeMode(at::musa::GetComputeModeFromCtx(l.scalar_type())),
        "SetComputeMode");
    CHECK_MUDNN_STATUS(b_mm.SetTranspose(trans_l, trans_r), "SetTranspose");
    CHECK_MUDNN_STATUS(b_mm.Run(h, rst, lmt, rmt, InternalMemAlloc), "Run");
  } else {
    ::musa::dnn::MatMul mm;
    CHECK_MUDNN_STATUS(
        mm.SetComputeMode(at::musa::GetComputeModeFromCtx(l.scalar_type())),
        "SetComputeMode");
    CHECK_MUDNN_STATUS(mm.SetTranspose(trans_l, trans_r), "SetTranspose");
    // not support broadcast
    if (bias.has_value()) {
      TORCH_INTERNAL_ASSERT(bias->dim() == 1, "bias must be 1d tensor\n");
      auto bmt = CreateMUTensor(bias.value());
      CHECK_MUDNN_STATUS(mm.SetAlpha(alpha.to<double>()), "SetAlpha");
      CHECK_MUDNN_STATUS(mm.SetBeta(beta.to<double>()), "SetBeta");
      CHECK_MUDNN_STATUS(mm.SetGamma(gama.to<double>()), "SetGamma");
      CHECK_MUDNN_STATUS(
          mm.RunWithBiasAdd(h, rst, lmt, rmt, bmt, InternalMemAlloc),
          "RunWithBiasAdd");
    } else {
      CHECK_MUDNN_STATUS(mm.SetAlpha(alpha.to<double>()), "SetAlpha");
      CHECK_MUDNN_STATUS(mm.SetBeta(beta.to<double>()), "SetBeta");
      CHECK_MUDNN_STATUS(mm.Run(h, rst, lmt, rmt, InternalMemAlloc), "Run");
    }
  }
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
      "mat1 and mat2 must be a matrix and mat1_shape[1] must equal to "
      "mat2_shape[0]");
  if (self.numel() == out.size(1) && self.dim() == 1) {
    // MUDNN matmul only support 1D bias tensor.
    // RunWithBiasAdd: out = alpha * l * r + beta * out + gama * bias.
    MmCall(mat1, mat2, self, out, false, alpha, 0, beta);
  } else {
    // Run: out = alpha * l * r
    // if out binds with self, need backup self.
    Tensor self_ = out.is_same(self) ? self.clone() : self;
    MmCall(mat1, mat2, c10::nullopt, out, false, alpha);
    if (!beta.equal(0)) {
      out.add_(self_, beta);
    }
  }
  return out;
}

at::Tensor& AddMm_(
    at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  Tensor result = at::empty(
      {mat1.size(0), mat2.size(1)},
      self.options().memory_format(at::MemoryFormat::Contiguous));
  TORCH_CHECK(
      self.sizes() == result.sizes(),
      "AddMm_ requires self tensor has same shape as result");

  AddMmOut(self, mat1, mat2, beta, alpha, result);
  self.copy_(result);
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

  if (self.dim() == 0 && self.size(0) == vec.size(0)) {
    MmCall(mat, vec, self, out, false, alpha, 0, beta);
  } else {
    MmCall(mat, vec, c10::nullopt, out, false, alpha, 0, beta);
    if (!beta.equal(0)) {
      out.add_(self, beta);
    }
  }
  return out;
}

at::Tensor& AddMv_(
    at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  TORCH_CHECK(
      mat.dim() == 2 && vec.dim() == 1, "AddMv_ inputs dims don't match");
  TORCH_CHECK(mat.size(-1) == vec.size(0), "AddMv_ inputs shape dont' match");

  at::Tensor result = at::empty({mat.size(0)}, mat.options());
  AddMvOut(self, mat, vec, beta, alpha, result);
  self.copy_(result);

  return self;
}

at::Tensor AddMv(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  TORCH_CHECK(mat.dim() == 2 && vec.dim() == 1, "AddMv inputs dim don't match");
  TORCH_CHECK(mat.size(-1) == vec.size(0), "AddMv inputs shape dont' match");

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
  MmCall(self, mat2, c10::nullopt, out, false, 1, 0);
  return out;
}

Tensor Mm(const Tensor& self, const Tensor& mat2) {
  Tensor result = at::empty(
      {self.size(0), mat2.size(1)},
      self.options().memory_format(at::MemoryFormat::Contiguous));
  MmOut(self, mat2, result);
  return result;
}

Tensor& BmmOut(const Tensor& self, const Tensor& mat2, Tensor& out) {
  const auto device_guard = c10::musa::MUSAGuard(self.device());
  TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3, "self must be a 3D matrix");
  TORCH_CHECK(
      self.size(0) == mat2.size(0) && self.size(2) == mat2.size(1),
      "self_shape[0] must equal to mat2_shape[0], and self_shape[2] "
      "must equal to mat2_shape[1]");
  MmCall(self, mat2, c10::nullopt, out, true);
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
