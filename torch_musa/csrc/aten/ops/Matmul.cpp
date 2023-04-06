#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

// Restore disabled warnings
#pragma GCC diagnostic pop

namespace at {
namespace native {
namespace musa {

void BmmCall(const Tensor& l, const Tensor& r, Tensor& out) {
  muHandle h;
  bool trans_l = false;
  bool trans_r = false;
  auto lmt = CreateMUTensor(l);
  auto rmt = CreateMUTensor(r);
  auto rst = CreateMUTensor(out);

  ::musa::dnn::BatchMatMul b_mm;
  ConfigFormat(out, rst, true);
  CHECK_MUDNN_STATUS(b_mm.SetTranspose(trans_l, trans_r), "SetTranspose");
  CHECK_MUDNN_STATUS(b_mm.Run(h, rst, lmt, rmt), "Run");
}

Tensor& MmAlphaBetaOut(
    const Tensor& l,
    const Tensor& r,
    const double alpha,
    const double beta,
    Tensor& out) {
  muHandle h;
  auto dim_tensor1 = l.dim();
  auto dim_tensor2 = r.dim();

  auto contiguous_l = Contiguous(l);
  auto contiguous_r = Contiguous(r);

  if (dim_tensor1 == 4 || dim_tensor2 == 4) {
    if (dim_tensor1 == 4) {
      contiguous_l =
          contiguous_l.view({-1, contiguous_l.size(2), contiguous_l.size(3)});
    }
    if (dim_tensor2 == 4) {
      contiguous_r =
          contiguous_r.view({-1, contiguous_r.size(2), contiguous_r.size(3)});
    }
    auto contiguous_out = out.view({-1, out.size(2), out.size(3)});
    BmmCall(contiguous_l, contiguous_r, contiguous_out);
  } else if (dim_tensor1 == 3 || dim_tensor2 == 3) {
    // batch-mm
    BmmCall(contiguous_l, contiguous_r, out);
  } else {
    // when dim_tensor1 <= 2 &&  dim_tensor1 <= 2 ,call dot
    auto lmt = CreateMUTensor(contiguous_l);
    auto rmt = CreateMUTensor(contiguous_r);
    auto rst = CreateMUTensor(out);
    ConfigFormat(out, rst, true);
    if (alpha == 1 && beta == 0) {
      ::musa::dnn::Dot dot;
      CHECK_MUDNN_STATUS(dot.Run(h, rst, lmt, rmt), "Run");
    } else {
      ::musa::dnn::MatMul mm;
      CHECK_MUDNN_STATUS(mm.SetAlpha(alpha), "SetAlpha");
      CHECK_MUDNN_STATUS(mm.SetBeta(beta), "SetBeta");
      CHECK_MUDNN_STATUS(mm.Run(h, rst, lmt, rmt), "Run");
    }
  }
  return out;
}

at::Tensor& AddMmOut(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  TORCH_CHECK(
      mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
  TORCH_CHECK(
      mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
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
      self.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of Addmm only support Float32, but now it is ",
      self.scalar_type());
  TORCH_CHECK(
      mat1.scalar_type() == at::ScalarType::Float,
      "Dtype of mat1 tensor of Addmm only support Float32, but now it is ",
      mat1.scalar_type());
  TORCH_CHECK(
      mat2.scalar_type() == at::ScalarType::Float,
      "Dtype of mat2 tensor of Addmm only support Float32, but now it is ",
      mat2.scalar_type());

  out.resize_({mat1.sizes()[0], mat2.sizes()[1]});
  auto expand_self =
      expand_size(self, {mat1.sizes()[0], mat2.sizes()[1]}, "addmm_out");
  out.copy_(*expand_self);
  // only support float32 now
  AT_DISPATCH_ALL_MTGPU_TYPES_AND_HALF(self.scalar_type(), "add_mm", [&] {
    auto beta_value = beta.to<scalar_t>();
    auto alpha_value = alpha.to<scalar_t>();
    MmAlphaBetaOut(
        mat1,
        mat2,
        static_cast<double>(alpha_value),
        static_cast<double>(beta_value),
        out);
  });
  return out;
}

at::Tensor AddMm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  auto result = at::empty({0}, self.options());
  AddMmOut(self, mat1, mat2, beta, alpha, result);
  return result;
}

Tensor Mm(const Tensor& self, const Tensor& mat2) {
  TORCH_CHECK(
      self.dim() == 2 && mat2.dim() == 2 && self.size(1) == mat2.size(0),
      "self must be a matrix and self_shape[1] must equal to mat2_shape[0]");
  Tensor result = empty_mtgpu(
      {self.size(0), mat2.size(1)},
      self.scalar_type(),
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  auto contiguous_l = Contiguous(self);
  auto contiguous_r = Contiguous(mat2);
  BmmCall(contiguous_l, contiguous_r, result);
  return result;
}

Tensor& MmOut(const Tensor& self, const Tensor& mat2, Tensor& out) {
  TORCH_CHECK(
      self.dim() == 2 && mat2.dim() == 2 && self.size(1) == mat2.size(0),
      "self and mat2 must be a matrix and self_shape[1] must equal to "
      "mat2_shape[0]");
  auto contiguous_l = Contiguous(self);
  auto contiguous_r = Contiguous(mat2);
  BmmCall(contiguous_l, contiguous_r, out);
  return out;
}

Tensor Bmm(const Tensor& self, const Tensor& mat2) {
  TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3, "self must be a 3D matrix");
  TORCH_CHECK(
      self.size(0) == mat2.size(0) && self.size(2) == mat2.size(1),
      "self_shape[0] must equal to mat2_shape[0], and self_shape[2] "
      "must equal to mat2_shape[1]");
  Tensor result = empty_mtgpu(
      {self.size(0), self.size(1), mat2.size(2)},
      self.scalar_type(),
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  auto l_ = Contiguous(self);
  auto r_ = Contiguous(mat2);
  BmmCall(l_, r_, result);
  return result;
}

Tensor& BmmOut(const Tensor& self, const Tensor& mat2, Tensor& out) {
  TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3, "self must be a 3D matrix");
  TORCH_CHECK(
      self.size(0) == mat2.size(0) && self.size(2) == mat2.size(1),
      "self_shape[0] must equal to mat2_shape[0], and self_shape[2] "
      "must equal to mat2_shape[1]");
  auto l_ = Contiguous(self);
  auto r_ = Contiguous(mat2);
  BmmCall(l_, r_, out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("addmm.out", &AddMmOut);
  m.impl("addmm", &AddMm);
  m.impl("mm", &Mm);
  m.impl("mm.out", &MmOut);
  m.impl("bmm", &Bmm);
  m.impl("bmm.out", &BmmOut);
}

} // namespace musa
} // namespace native
} // namespace at
