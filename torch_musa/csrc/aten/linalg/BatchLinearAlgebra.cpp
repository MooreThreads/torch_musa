#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <utility>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/musa/MUSAContext.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#endif

#include "torch_musa/csrc/aten/linalg/BatchLinearAlgebraLib.h"
#include "torch_musa/csrc/aten/linalg/BatchLinearAlgebraLibBlas.h"

namespace at::native {

/*
NOTE(@mt-ai): PyTorch is using Magma, cusolver, cublas and custom kernel for
  these linalg kernels, while we only use musolver for now.
*/

static void cholesky_kernel(
    const Tensor& input,
    const Tensor& info,
    bool upper) {
  at::musa::cholesky_helper_musolver(input, upper, info);
}

static void lu_solve_kernel(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& B,
    TransposeType trans) {
  // Trivial case. Remove it once `torch.solve` is removed, as linalg.solve
  // already shortcuts this case
  if (B.numel() == 0) {
    return;
  }

  auto b = at::native::batchCount(B);
  auto n = LU.size(-2);
  auto k = B.size(-1);

  // Computes X = U^{-1}L^{-1}P^T B via triangular solves
  at::musa::lu_solve_looped_musolver(LU, pivots, B, trans);
}

void geqrf_kernel_musa(const Tensor& input, const Tensor& tau) {
  auto geqrf_musolver_backend = [](const Tensor& input, const Tensor& tau) {
    return at::musa::geqrf_musolver(input, tau);
  };
  return geqrf_musolver_backend(input, tau);
}

Tensor& orgqr_kernel_impl(Tensor& result, const Tensor& tau) {
  return at::musa::orgqr_helper_musolver(result, tau); // musolver
}

void triangular_solve_kernel(
    const Tensor& A,
    const Tensor& B,
    bool left,
    bool upper,
    TransposeType transpose,
    bool unitriangular) {
  // For batches smaller than 8 and matrix sizes larger than 64x64 cuBLAS
  // forloop is faster than batched version
  if (batchCount(A) <= 8 && A.size(-1) >= 64) {
    triangular_solve_mublas(A, B, left, upper, transpose, unitriangular);
  } else {
    triangular_solve_batched_mublas(
        A, B, left, upper, transpose, unitriangular);
  }
}

void ldl_factor_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
  // musolver 1.4.0 doesnot support sytrf buffersize, using cpu implementation
  auto LD_ = LD.to(Device(at::kCPU));
  auto pivots_ = pivots.to(Device(at::kCPU));
  auto info_ = info.to(Device(at::kCPU));

  ldl_factor_stub(LD_.device().type(), LD_, pivots_, info_, upper, hermitian);
  // call ldl_factor_stub that fills the result tensors
  Tensor& modified_LD = const_cast<Tensor&>(LD);
  Tensor& modified_pivots = const_cast<Tensor&>(pivots);
  Tensor& modified_info = const_cast<Tensor&>(info);
  modified_LD.copy_(LD_);
  modified_pivots.copy_(pivots_);
  modified_info.copy_(info_);
}

void ldl_solve_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool upper,
    bool hermitian) {
  // todo: musolver support Xsytrs, 1.4.0 doesn't support
  auto LD_ = LD.to(Device(at::kCPU));
  auto pivots_ = pivots.to(Device(at::kCPU));
  auto B_ = B.to(Device(at::kCPU));
  ldl_solve_stub(LD_.device().type(), LD_, pivots_, B_, upper, hermitian);

  Tensor& modified_B = const_cast<Tensor&>(B);
  modified_B.copy_(B_);
}

void linalg_eigh_kernel(
    const Tensor& eigenvalues,
    const Tensor& eigenvectors,
    const Tensor& infos,
    bool upper,
    bool compute_eigenvectors) {
  // todo: musolver support syevj, 1.4.0 doesn't support syevj_buffersize, and
  // doesnot support complex dtype syevj
  auto eigenvalues_ = eigenvalues.to(Device(at::kCPU));
  auto eigenvectors_ = eigenvectors.to(Device(at::kCPU));
  auto infos_ = infos.to(Device(at::kCPU));

  linalg_eigh_stub(
      eigenvalues_.device().type(),
      eigenvalues_,
      eigenvectors_,
      infos_,
      upper,
      compute_eigenvectors);

  Tensor& modified_eigenvalues = const_cast<Tensor&>(eigenvalues);
  Tensor& modified_eigenvectors = const_cast<Tensor&>(eigenvectors);
  Tensor& modified_infos = const_cast<Tensor&>(infos);
  modified_eigenvalues.copy_(eigenvalues_);
  modified_eigenvectors.copy_(eigenvectors_);
  modified_infos.copy_(infos_);
}

void svd_kernel(
    const Tensor& A,
    const bool full_matrices,
    const bool compute_uv,
    const std::optional<std::string_view>& driver,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh,
    const Tensor& info) {
  // todo: musolver support gesvdj_buffersize, 1.4.0 doesn't support
  // gesvdj_buffersize
  auto A_ = A.to(Device(at::kCPU));
  auto U_ = U.to(Device(at::kCPU));
  auto S_ = S.to(Device(at::kCPU));
  auto Vh_ = Vh.to(Device(at::kCPU));
  auto info_ = info.to(Device(at::kCPU));

  svd_stub(
      A.device().type(),
      A_,
      full_matrices,
      compute_uv,
      driver,
      U_,
      S_,
      Vh_,
      info_);

  Tensor& modified_A = const_cast<Tensor&>(A);
  Tensor& modified_U = const_cast<Tensor&>(U);
  Tensor& modified_S = const_cast<Tensor&>(S);
  Tensor& modified_Vh = const_cast<Tensor&>(Vh);
  Tensor& modified_info = const_cast<Tensor&>(info);
  modified_A.copy_(A_);
  modified_U.copy_(U_);
  modified_S.copy_(S_);
  modified_Vh.copy_(Vh_);
  modified_info.copy_(info_);
}

void ormqr_kernel_impl(
    const Tensor& input,
    const Tensor& tau,
    const Tensor& other,
    bool left,
    bool transpose) {
  at::musa::ormqr_helper_musolver(
      input, tau, other, left, transpose); // musolver
}

REGISTER_MUSA_DISPATCH(ldl_factor_stub, &ldl_factor_kernel)
REGISTER_MUSA_DISPATCH(geqrf_stub, &geqrf_kernel_musa)
REGISTER_MUSA_DISPATCH(orgqr_stub, &orgqr_kernel_impl)
REGISTER_MUSA_DISPATCH(ormqr_stub, &ormqr_kernel_impl);
REGISTER_MUSA_DISPATCH(cholesky_stub, &cholesky_kernel)
REGISTER_MUSA_DISPATCH(lu_solve_stub, &lu_solve_kernel)
REGISTER_MUSA_DISPATCH(triangular_solve_stub, &triangular_solve_kernel)
REGISTER_MUSA_DISPATCH(ldl_solve_stub, &ldl_solve_kernel)
REGISTER_MUSA_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel)
REGISTER_MUSA_DISPATCH(svd_stub, &svd_kernel)

} // namespace at::native