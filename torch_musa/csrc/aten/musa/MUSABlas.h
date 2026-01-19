#ifndef TORCH_MUSA_CSRC_ATEN_MUSA_MUSABLAS_H_
#define TORCH_MUSA_CSRC_ATEN_MUSA_MUSABLAS_H_

#include <ATen/OpMathType.h>
#include <internal/mublas_types.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"

namespace at {
namespace musa {
namespace blas {

#define MUSABLAS_GEMM_ARGTYPES(Dtype)                                     \
  char transa, char transb, int64_t m, int64_t n, int64_t k,              \
      at::opmath_type<Dtype> alpha, const Dtype *a, int64_t lda,          \
      const Dtype *b, int64_t ldb, at::opmath_type<Dtype> beta, Dtype *c, \
      int64_t ldc

template <typename Dtype>
inline void gemm(MUSABLAS_GEMM_ARGTYPES(Dtype)) {
  AT_ERROR("at::musa::blas::gemm: not implemented for ", typeid(Dtype).name());
}

template <>
void gemm<double>(MUSABLAS_GEMM_ARGTYPES(double));
template <>
void gemm<float>(MUSABLAS_GEMM_ARGTYPES(float));
template <>
void gemm<c10::complex<double>>(MUSABLAS_GEMM_ARGTYPES(c10::complex<double>));
template <>
void gemm<c10::complex<float>>(MUSABLAS_GEMM_ARGTYPES(c10::complex<float>));
template <>
void gemm<at::Half>(MUSABLAS_GEMM_ARGTYPES(at::Half));
template <>
void gemm<at::BFloat16>(MUSABLAS_GEMM_ARGTYPES(at::BFloat16));

#define MUSABLAS_BGEMM_ARGTYPES(Dtype)                                     \
  char transa, char transb, int64_t m, int64_t n, int64_t k,               \
      at::opmath_type<Dtype> alpha, const Dtype *a, int64_t lda,           \
      int64_t stridea, const Dtype *b, int64_t ldb, int64_t strideb,       \
      at::opmath_type<Dtype> beta, Dtype *c, int64_t ldc, int64_t stridec, \
      int64_t num_batches

template <typename Dtype>
inline void bgemm(MUSABLAS_BGEMM_ARGTYPES(Dtype)) {
  AT_ERROR("at::musa::blas::bgemm: not implemented for ", typeid(Dtype).name());
}

template <>
void bgemm<double>(MUSABLAS_BGEMM_ARGTYPES(double));
template <>
void bgemm<float>(MUSABLAS_BGEMM_ARGTYPES(float));
template <>
void bgemm<c10::complex<double>>(MUSABLAS_BGEMM_ARGTYPES(c10::complex<double>));
template <>
void bgemm<c10::complex<float>>(MUSABLAS_BGEMM_ARGTYPES(c10::complex<float>));
template <>
void bgemm<at::Half>(MUSABLAS_BGEMM_ARGTYPES(at::Half));
template <>
void bgemm<at::BFloat16>(MUSABLAS_BGEMM_ARGTYPES(at::BFloat16));

#define MUSABLAS_VDOT_ARGTYPES(Dtype) \
  int n, const Dtype *x, int incx, const Dtype *y, int incy, Dtype *result

template <typename Dtype>
inline void vdot(MUSABLAS_VDOT_ARGTYPES(Dtype)) {
  AT_ERROR("at::musa::blas::vdot: not implemented for ", typeid(Dtype).name());
}

template <>
void vdot<c10::complex<float>>(MUSABLAS_VDOT_ARGTYPES(c10::complex<float>));
template <>
void vdot<c10::complex<double>>(MUSABLAS_VDOT_ARGTYPES(c10::complex<double>));

#define MUSABLAS_TRSM_ARGTYPES(Dtype)                                  \
  mublasHandle_t handle, mublasSideMode_t side, mublasFillMode_t uplo, \
      mublasOperation_t trans, mublasDiagType_t diag, int m, int n,    \
      const Dtype *alpha, const Dtype *A, int lda, Dtype *B, int ldb

template <typename Dtype>
inline void trsm(MUSABLAS_TRSM_ARGTYPES(Dtype)) {
  static_assert(
      false && sizeof(Dtype), "at::musa::blas::trsm: not implemented");
}

template <>
void trsm<float>(MUSABLAS_TRSM_ARGTYPES(float));
template <>
void trsm<double>(MUSABLAS_TRSM_ARGTYPES(double));
template <>
void trsm<c10::complex<float>>(MUSABLAS_TRSM_ARGTYPES(c10::complex<float>));
template <>
void trsm<c10::complex<double>>(MUSABLAS_TRSM_ARGTYPES(c10::complex<double>));

#define MUSABLAS_TRSM_BATCHED_ARGTYPES(Dtype)                          \
  mublasHandle_t handle, mublasSideMode_t side, mublasFillMode_t uplo, \
      mublasOperation_t trans, mublasDiagType_t diag, int m, int n,    \
      const Dtype *alpha, Dtype *A[], int lda, Dtype *B[], int ldb,    \
      int batchCount

template <typename Dtype>
inline void trsmBatched(MUSABLAS_TRSM_BATCHED_ARGTYPES(Dtype)) {
  static_assert(
      false && sizeof(Dtype), "at::musa::blas::trsmBatched: not implemented");
}

template <>
void trsmBatched<float>(MUSABLAS_TRSM_BATCHED_ARGTYPES(float));
template <>
void trsmBatched<double>(MUSABLAS_TRSM_BATCHED_ARGTYPES(double));
template <>
void trsmBatched<c10::complex<float>>(
    MUSABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<float>));
template <>
void trsmBatched<c10::complex<double>>(
    MUSABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<double>));

} // namespace blas
} // namespace musa
} // namespace at
#endif // TORCH_MUSA_CSRC_ATEN_MUSA_MUSABLAS_H_
