#ifndef TORCH_MUSA_CSRC_ATEN_LINALG_MUSASOLVER_H_
#define TORCH_MUSA_CSRC_ATEN_LINALG_MUSASOLVER_H_

#include <musolver.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"

namespace at::musa::solver {

#define MUSASOLVER_GETRF_ARGTYPES(Dtype) \
  mublasHandle_t handle, int m, int n, Dtype *dA, int ldda, int *ipiv, int *info

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 5010)
#define MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(scalar_t) \
  int m, int n, size_t *lwork

#define MUSASOLVER_GEQRF_ARGTYPES(scalar_t)                                   \
  mublasHandle_t handle, int m, int n, scalar_t *dA, int lda, scalar_t *ipiv, \
      void *buffer

#define MUSASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(Dtype) \
  int m, int n, int k, size_t *lwork

#define MUSASOLVER_ORGQR_ARGTYPES(Dtype)                                  \
  mublasHandle_t handle, const int m, const int n, const int k, Dtype *A, \
      const int lda, Dtype *ipiv, void *buffer

#define MUSASOLVER_ORMQR_ARGTYPES(Dtype)                                    \
  mublasHandle_t handle, const mublasSideMode_t side,                       \
      const mublasOperation_t trans, const int m, const int n, const int k, \
      Dtype *A, const int lda, Dtype *ipiv, Dtype *C, const int ldc

#else
#define MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(scalar_t) \
  mublasHandle_t handle, int m, int n, scalar_t *A, int lda, int *lwork

#define MUSASOLVER_GEQRF_ARGTYPES(scalar_t)                                 \
  mublasHandle_t handle, int m, int n, scalar_t *A, int lda, scalar_t *tau, \
      scalar_t *work, int lwork, int *devInfo

#define MUSASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(Dtype)                    \
  mublasHandle_t handle, int m, int n, int k, const Dtype *A, int lda, \
      const Dtype *tau, int *lwork

#define MUSASOLVER_ORGQR_ARGTYPES(Dtype)                         \
  mublasHandle_t handle, int m, int n, int k, Dtype *A, int lda, \
      const Dtype *tau, Dtype *work, int lwork, int *devInfo

#define MUSASOLVER_ORMQR_BUFFERSIZE_ARGTYPES_REAL(Dtype)                       \
  mublasHandle_t handle, const mublasSideMode_t side, mublasOperation_t trans, \
      int m, int n, int k, Dtype *A, int lda, Dtype *tau, Dtype *C, int ldc,   \
      int *lwork

#define MUSASOLVER_ORMQR_ARGTYPES(Dtype)                                    \
  mublasHandle_t handle, const mublasSideMode_t side,                       \
      const mublasOperation_t trans, const int m, const int n, const int k, \
      Dtype *A, const int lda, Dtype *ipiv, Dtype *C, const int ldc,        \
      Dtype *buffer, int bufferSize, int *devInfo

#endif

template <class Dtype>
void getrf(MUSASOLVER_GETRF_ARGTYPES(Dtype)) {
  static_assert(
      false && sizeof(Dtype), "at::musa::solver::getrf: not implemented");
}
template <>
void getrf<double>(MUSASOLVER_GETRF_ARGTYPES(double));
template <>
void getrf<float>(MUSASOLVER_GETRF_ARGTYPES(float));
template <>
void getrf<c10::complex<double>>(
    MUSASOLVER_GETRF_ARGTYPES(c10::complex<double>));
template <>
void getrf<c10::complex<float>>(MUSASOLVER_GETRF_ARGTYPES(c10::complex<float>));

#define MUSASOLVER_POTRS_ARGTYPES(Dtype)                                   \
  mublasHandle_t handle, mublasFillMode_t uplo, int n, int nrhs, Dtype *A, \
      int lda, Dtype *B, int ldb, int *devInfo

template <class Dtype>
void potrs(MUSASOLVER_POTRS_ARGTYPES(Dtype)) {
  static_assert(
      false && sizeof(Dtype), "at::musa::solver::potrs: not implemented");
}
template <>
void potrs<float>(MUSASOLVER_POTRS_ARGTYPES(float));
template <>
void potrs<double>(MUSASOLVER_POTRS_ARGTYPES(double));
template <>
void potrs<c10::complex<float>>(MUSASOLVER_POTRS_ARGTYPES(c10::complex<float>));
template <>
void potrs<c10::complex<double>>(
    MUSASOLVER_POTRS_ARGTYPES(c10::complex<double>));

#define MUSASOLVER_GETRS_ARGTYPES(Dtype)                                 \
  mublasHandle_t handle, int n, int nrhs, Dtype *dA, int lda, int *ipiv, \
      Dtype *ret, int ldb, int *info, mublasOperation_t trans

template <class Dtype>
void getrs(MUSASOLVER_GETRS_ARGTYPES(Dtype)) {
  static_assert(
      false && sizeof(Dtype), "at::musa::solver::getrs: not implemented");
}
template <>
void getrs<float>(MUSASOLVER_GETRS_ARGTYPES(float));
template <>
void getrs<double>(MUSASOLVER_GETRS_ARGTYPES(double));
template <>
void getrs<c10::complex<double>>(
    MUSASOLVER_GETRS_ARGTYPES(c10::complex<double>));
template <>
void getrs<c10::complex<float>>(MUSASOLVER_GETRS_ARGTYPES(c10::complex<float>));

#define MUSASOLVER_POTRF_BATCHED_ARGTYPES(Dtype)                           \
  mublasHandle_t handle, mublasFillMode_t uplo, int n, Dtype **A, int lda, \
      int *info, int batchSize

template <class Dtype>
void potrfBatched(MUSASOLVER_POTRF_BATCHED_ARGTYPES(Dtype)) {
  static_assert(
      false && sizeof(Dtype),
      "at::musa::solver::potrfBatched: not implemented");
}
template <>
void potrfBatched<float>(MUSASOLVER_POTRF_BATCHED_ARGTYPES(float));
template <>
void potrfBatched<double>(MUSASOLVER_POTRF_BATCHED_ARGTYPES(double));
template <>
void potrfBatched<c10::complex<float>>(
    MUSASOLVER_POTRF_BATCHED_ARGTYPES(c10::complex<float>));
template <>
void potrfBatched<c10::complex<double>>(
    MUSASOLVER_POTRF_BATCHED_ARGTYPES(c10::complex<double>));

template <class scalar_t>
void geqrf_bufferSize(MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(scalar_t)) {
  static_assert(
      false && sizeof(scalar_t),
      "at::musa::solver::geqrf_bufferSize: not implemented");
}
template <>
void geqrf_bufferSize<float>(MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(float));
template <>
void geqrf_bufferSize<double>(MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(double));
template <>
void geqrf_bufferSize<c10::complex<float>>(
    MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(c10::complex<float>));
template <>
void geqrf_bufferSize<c10::complex<double>>(
    MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(c10::complex<double>));

template <class scalar_t>
void geqrf(MUSASOLVER_GEQRF_ARGTYPES(scalar_t)) {
  static_assert(
      false && sizeof(scalar_t), "at::musa::solver::geqrf: not implemented");
}
template <>
void geqrf<float>(MUSASOLVER_GEQRF_ARGTYPES(float));
template <>
void geqrf<double>(MUSASOLVER_GEQRF_ARGTYPES(double));
template <>
void geqrf<c10::complex<float>>(MUSASOLVER_GEQRF_ARGTYPES(c10::complex<float>));
template <>
void geqrf<c10::complex<double>>(
    MUSASOLVER_GEQRF_ARGTYPES(c10::complex<double>));

template <class Dtype>
void orgqr_buffersize(MUSASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(Dtype)) {
  static_assert(
      false && sizeof(Dtype),
      "at::musa::solver::orgqr_buffersize: not implemented");
}
template <>
void orgqr_buffersize<float>(MUSASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(float));
template <>
void orgqr_buffersize<double>(MUSASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(double));
template <>
void orgqr_buffersize<c10::complex<float>>(
    MUSASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(c10::complex<float>));
template <>
void orgqr_buffersize<c10::complex<double>>(
    MUSASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(c10::complex<double>));

template <class Dtype>
void orgqr(MUSASOLVER_ORGQR_ARGTYPES(Dtype)) {
  static_assert(
      false && sizeof(Dtype), "at::musa::solver::orgqr: not implemented");
}
template <>
void orgqr<float>(MUSASOLVER_ORGQR_ARGTYPES(float));
template <>
void orgqr<double>(MUSASOLVER_ORGQR_ARGTYPES(double));
template <>
void orgqr<c10::complex<float>>(MUSASOLVER_ORGQR_ARGTYPES(c10::complex<float>));
template <>
void orgqr<c10::complex<double>>(
    MUSASOLVER_ORGQR_ARGTYPES(c10::complex<double>));

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 5000)
template <class Dtype>
void ormqr(MUSASOLVER_ORMQR_ARGTYPES(Dtype)) {
  static_assert(
      false && sizeof(Dtype), "at::musa::solver::ormqr: not implemented");
}

template <>
void ormqr<float>(MUSASOLVER_ORMQR_ARGTYPES(float));

template <>
void ormqr<double>(MUSASOLVER_ORMQR_ARGTYPES(double));

template <>
void ormqr<c10::complex<float>>(MUSASOLVER_ORMQR_ARGTYPES(c10::complex<float>));

template <>
void ormqr<c10::complex<double>>(
    MUSASOLVER_ORMQR_ARGTYPES(c10::complex<double>));

#else
template <class Dtype>
void ormqr_buffersize(MUSASOLVER_ORMQR_BUFFERSIZE_ARGTYPES_REAL(Dtype)) {
  static_assert(
      false && sizeof(Dtype),
      "at::musa::solver::ormqr_buffersize: not implemented");
}

template <>
void ormqr_buffersize<float>(MUSASOLVER_ORMQR_BUFFERSIZE_ARGTYPES_REAL(float));

template <>
void ormqr_buffersize<double>(
    MUSASOLVER_ORMQR_BUFFERSIZE_ARGTYPES_REAL(double));

template <class Dtype>
void ormqr(MUSASOLVER_ORMQR_ARGTYPES(Dtype)) {
  static_assert(
      false && sizeof(Dtype), "at::musa::solver::ormqr: not implemented");
}

template <>
void ormqr<float>(MUSASOLVER_ORMQR_ARGTYPES(float));

template <>
void ormqr<double>(MUSASOLVER_ORMQR_ARGTYPES(double));

template <>
void ormqr<c10::complex<float>>(MUSASOLVER_ORMQR_ARGTYPES(c10::complex<float>));

template <>
void ormqr<c10::complex<double>>(
    MUSASOLVER_ORMQR_ARGTYPES(c10::complex<double>));

#endif

#define MUSASOLVER_POTRS_BATCHED_ARGTYPES(Dtype)                     \
  mublasHandle_t handle, mublasFillMode_t uplo, int n, int nrhs,     \
      Dtype *Aarray[], int lda, Dtype *Barray[], int ldb, int *info, \
      int batchSize

template <class Dtype>
void potrsBatched(MUSASOLVER_POTRS_BATCHED_ARGTYPES(Dtype)) {
  static_assert(
      false && sizeof(Dtype),
      "at::musa::solver::potrsBatched: not implemented");
}
template <>
void potrsBatched<float>(MUSASOLVER_POTRS_BATCHED_ARGTYPES(float));
template <>
void potrsBatched<double>(MUSASOLVER_POTRS_BATCHED_ARGTYPES(double));
template <>
void potrsBatched<c10::complex<float>>(
    MUSASOLVER_POTRS_BATCHED_ARGTYPES(c10::complex<float>));
template <>
void potrsBatched<c10::complex<double>>(
    MUSASOLVER_POTRS_BATCHED_ARGTYPES(c10::complex<double>));

} // namespace at::musa::solver
#endif // TORCH_MUSA_CSRC_ATEN_LINALG_MUSASOLVER_H_
