#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include <mudnn.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/ops/musa/musa_ops.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
namespace at {
namespace musa {

at::Tensor Dot(const at::Tensor& l, const at::Tensor& r) {
  c10::musa::MUSAGuard device_guard(l.device());
  muHandle& h = GetMudnnHandle();
  Tensor out = at::empty(
      {1, 1}, l.options().memory_format(at::MemoryFormat::Contiguous));
  auto rst = CreateMUTensor(out);
  // TODO : (lms)view out, l and r to fix the mudnn's convention
  // https://github.mthreads.com/mthreads/muDNN/issues/304
  Tensor contiguous_l = l.contiguous().view(c10::IntArrayRef{1, l.size(0)});
  Tensor contiguous_r = r.contiguous().view(c10::IntArrayRef{l.size(0), 1});
  auto lmt = CreateMUTensor(contiguous_l);
  auto rmt = CreateMUTensor(contiguous_r);

  ::musa::dnn::Dot op;
  op.SetComputeMode(::musa::dnn::Dot::ComputeMode::TENSOR);
  CHECK_MUDNN_STATUS(op.Run(h, rst, lmt, rmt, InternalMemAlloc), "Run")
  return out.squeeze();
}

void MmCall(
    const Tensor& l,
    const Tensor& r,
    const Tensor& bias,
    Tensor& out,
    bool is_batch = false,
    const at::Scalar& alpha = 1,
    const at::Scalar beta = 0,
    const at::Scalar gama = 0) {
  c10::musa::MUSAGuard device_guard(l.device());
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
    CHECK_MUDNN_STATUS(b_mm.SetTranspose(trans_l, trans_r), "SetTranspose");
    CHECK_MUDNN_STATUS(b_mm.Run(h, rst, lmt, rmt, InternalMemAlloc), "Run");
  } else {
    ::musa::dnn::MatMul mm;
    CHECK_MUDNN_STATUS(mm.SetTranspose(trans_l, trans_r), "SetTranspose");
    // not support broadcast
    if (beta.equal(0) && !gama.equal(0) && bias.numel() == out.size(1) &&
        bias.dim() == 1) {
      auto contiguous_bias = bias.contiguous();
      auto bmt = CreateMUTensor(contiguous_bias);
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
  TORCH_CHECK(
      mat1.dim() == 2 && mat2.dim() == 2 && mat1.size(1) == mat2.size(0),
      "mat1 and mat2 must be a matrix and mat1_shape[1] must equal to "
      "mat2_shape[0]");
  // normal case
  // RunWithBiasAdd: out = alpha * l * r + beta * out + gama * bias, not support
  // broadcast now
  if (!beta.equal(0) && self.numel() == out.size(1) && self.dim() == 1) {
    MmCall(mat1, mat2, self, out, false, alpha, 0, beta);
  } else {
    // Run: out = beta * out + gama * bias
    out.zero_();
    out.add_(self);
    MmCall(mat1, mat2, out, out, false, alpha, beta, 0);
  }
  return out;
}

at::Tensor& AddMvOut(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  TORCH_CHECK(
      mat1.dim() == 2 && mat2.dim() == 1 && mat1.size(1) == mat2.size(0),
      "mat1 and mat2 must be a matrix and mat1_shape[1] must equal to "
      "mat2_shape[0]");
  std::vector<int64_t> sz;
  sz.push_back(mat2.size(0));
  sz.push_back(1);
  const at::Tensor& vec = mat2.view(sz);
  // TODO:(lms) to avoid mudnn failing in gemv
  // https://github.mthreads.com/mthreads/muDNN/issues/304
  out = out.unsqueeze(-1);
  AddMmOut(self, mat1, vec, beta, alpha, out);
  return out.squeeze_();
}

at::Tensor AddMm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  c10::musa::MUSAGuard(self.device());
  Tensor result = at::empty(
      {mat1.size(0), mat2.size(1)},
      self.options().memory_format(at::MemoryFormat::Contiguous));
  AddMmOut(self, mat1, mat2, beta, alpha, result);
  return result;
}

Tensor& MmOut(const Tensor& self, const Tensor& mat2, Tensor& out) {
  TORCH_CHECK(
      self.dim() == 2 && mat2.dim() == 2 && self.size(1) == mat2.size(0),
      "self and mat2 must be a matrix and self_shape[1] must equal to "
      "mat2_shape[0]");
  MmCall(self, mat2, out, out, false, 1, 0);
  return out;
}

Tensor Mm(const Tensor& self, const Tensor& mat2) {
  c10::musa::MUSAGuard(self.device());
  Tensor result = at::empty(
      {self.size(0), mat2.size(1)},
      self.options().memory_format(at::MemoryFormat::Contiguous));
  MmOut(self, mat2, result);
  return result;
}

Tensor& BmmOut(const Tensor& self, const Tensor& mat2, Tensor& out) {
  TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3, "self must be a 3D matrix");
  TORCH_CHECK(
      self.size(0) == mat2.size(0) && self.size(2) == mat2.size(1),
      "self_shape[0] must equal to mat2_shape[0], and self_shape[2] "
      "must equal to mat2_shape[1]");
  MmCall(self, mat2, out, out, true);
  return out;
}
Tensor Bmm(const Tensor& self, const Tensor& mat2) {
  c10::musa::MUSAGuard(self.device());
  Tensor result = at::empty(
      {self.size(0), self.size(1), mat2.size(2)},
      self.options().memory_format(at::MemoryFormat::Contiguous));
  BmmOut(self, mat2, result);
  return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("dot", &Dot);
  m.impl("addmv.out", &AddMvOut);
  m.impl("addmm.out", &AddMmOut);
  m.impl("addmm", &AddMm);
  m.impl("mm", &Mm);
  m.impl("mm.out", &MmOut);
  m.impl("bmm", &Bmm);
  m.impl("bmm.out", &BmmOut);
  m.impl("baddbmm", &Baddbmm);
}

} // namespace musa
} // namespace at
