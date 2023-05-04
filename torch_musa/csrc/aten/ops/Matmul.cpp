#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include <mudnn.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace native {
namespace musa {
// To judge whether the matrix is ​​transposed, the following two conditions
// need to be met
// 1. m * n matrix, matrix.stride(0) == 1, matrix.stride(1) == m
// 2. the origin matrix(untransposed matrix) should be contiguous
bool IsTranspose(const Tensor& mat, bool is_batch) {
  int batch_index = is_batch ? 1 : 0;
  if (mat.stride(0 + batch_index) == 1 &&
      mat.stride(1 + batch_index) == mat.size(0 + batch_index) &&
      IsContiguous(mat.transpose(0 + batch_index, 1 + batch_index))) {
    return true;
  } else {
    return false;
  }
}

void BmmCall(const Tensor& l, const Tensor& r, Tensor& out, bool is_batch) {
  muHandle h;
  bool trans_l = IsTranspose(l, is_batch);
  bool trans_r = IsTranspose(r, is_batch);
  int batch_index = is_batch ? 1 : 0;
  // if IsTranspose(mat) is True, we don't need to clone to permutate memory
  Tensor contiguous_l;
  Tensor contiguous_r;
  auto lmt = trans_l
      ? CreateMUTensor(l.transpose(0 + batch_index, 1 + batch_index), true)
      : CreateMUTensor(Contiguous(l, contiguous_l));
  auto rmt = trans_r
      ? CreateMUTensor(r.transpose(0 + batch_index, 1 + batch_index), true)
      : CreateMUTensor(Contiguous(r, contiguous_r));
  auto rst = CreateMUTensor(out);
  ConfigFormat(out, rst, true);
  ::musa::dnn::BatchMatMul b_mm;
  CHECK_MUDNN_STATUS(b_mm.SetTranspose(trans_l, trans_r), "SetTranspose");
  CHECK_MUDNN_STATUS(b_mm.Run(h, rst, lmt, rmt), "Run");
}

void MmCall(
    const Tensor& l,
    const Tensor& r,
    Tensor& out,
    double alpha,
    double beta) {
  muHandle h;
  bool trans_l = IsTranspose(l, false);
  bool trans_r = IsTranspose(r, false);
  Tensor contiguous_l;
  Tensor contiguous_r;
  auto lmt = trans_l ? CreateMUTensor(l.t(), true)
                     : CreateMUTensor(Contiguous(l, contiguous_l));
  auto rmt = trans_r ? CreateMUTensor(r.t(), true)
                     : CreateMUTensor(Contiguous(r, contiguous_r));
  auto rst = CreateMUTensor(out);
  ConfigFormat(out, rst, true);
  ::musa::dnn::MatMul matmul;
  CHECK_MUDNN_STATUS(matmul.SetTranspose(trans_l, trans_r), "SetTranspose");
  CHECK_MUDNN_STATUS(matmul.SetAlpha(alpha), "SetAlpha");
  CHECK_MUDNN_STATUS(matmul.SetBeta(beta), "SetBeta");
  CHECK_MUDNN_STATUS(matmul.Run(h, rst, lmt, rmt), "Run");
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

  if (dim_tensor1 == 4 || dim_tensor2 == 4) {
    auto view_l = l;
    auto view_r = r;
    if (dim_tensor1 == 4) {
      view_l = view_l.view({-1, view_l.size(2), view_l.size(3)});
    }
    if (dim_tensor2 == 4) {
      view_r = view_r.view({-1, view_r.size(2), view_r.size(3)});
    }
    auto view_out = out.view({-1, out.size(2), out.size(3)});
    BmmCall(view_l, view_r, view_out, false);
  } else if (dim_tensor1 == 3 || dim_tensor2 == 3) {
    // batch-mm
    BmmCall(l, r, out, true);
  } else {
    // when dim_tensor1 <= 2 &&  dim_tensor1 <= 2 ,call dot
    if (alpha == 1 && beta == 0) {
      ::musa::dnn::Dot dot;
      auto contiguous_out = Contiguous(out);
      auto rst = CreateMUTensor(contiguous_out);
      ConfigFormat(out, rst, true);
      auto contiguous_l = Contiguous(l);
      auto contiguous_r = Contiguous(r);
      auto lmt = CreateMUTensor(contiguous_l);
      auto rmt = CreateMUTensor(contiguous_r);
      CHECK_MUDNN_STATUS(dot.Run(h, rst, lmt, rmt), "Run");
    } else {
      MmCall(l, r, out, alpha, beta);
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

  out.zero_();
  out.add_(self);
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
  Tensor result = empty_mtgpu(
      {mat1.size(0), mat2.size(1)},
      self.scalar_type(),
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
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
  BmmCall(self, mat2, result, false);
  return result;
}

Tensor& MmOut(const Tensor& self, const Tensor& mat2, Tensor& out) {
  TORCH_CHECK(
      self.dim() == 2 && mat2.dim() == 2 && self.size(1) == mat2.size(0),
      "self and mat2 must be a matrix and self_shape[1] must equal to "
      "mat2_shape[0]");
  MmCall(self, mat2, out, 1, 1);
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
  BmmCall(self, mat2, result, true);
  return result;
}

Tensor& BmmOut(const Tensor& self, const Tensor& mat2, Tensor& out) {
  TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3, "self must be a 3D matrix");
  TORCH_CHECK(
      self.size(0) == mat2.size(0) && self.size(2) == mat2.size(1),
      "self_shape[0] must equal to mat2_shape[0], and self_shape[2] "
      "must equal to mat2_shape[1]");
  BmmCall(self, mat2, out, true);
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
