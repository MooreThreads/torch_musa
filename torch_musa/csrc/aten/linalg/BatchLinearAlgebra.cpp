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

REGISTER_MUSA_DISPATCH(cholesky_stub, &cholesky_kernel)
REGISTER_MUSA_DISPATCH(lu_solve_stub, &lu_solve_kernel);

} // namespace at::native