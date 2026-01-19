#ifndef BATCH_LINEAR_ALGEBRA_LIBBLAS_H
#define BATCH_LINEAR_ALGEBRA_LIBBLAS_H

#include <ATen/native/TransposeType.h>
#include <ATen/ops/scalar_tensor.h>

namespace at::native {
void triangular_solve_mublas(
    const Tensor& A,
    const Tensor& B,
    bool left,
    bool upper,
    TransposeType transpose,
    bool unitriangular);
void triangular_solve_batched_mublas(
    const Tensor& A,
    const Tensor& B,
    bool left,
    bool upper,
    TransposeType transpose,
    bool unitriangular);
} // namespace at::native

#endif // BATCH_LINEAR_ALGEBRA_LIBBLAS_H