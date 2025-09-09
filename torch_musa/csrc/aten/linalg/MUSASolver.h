#ifndef TORCH_MUSA_CSRC_ATEN_LINALG_MUSASOLVER_H_
#define TORCH_MUSA_CSRC_ATEN_LINALG_MUSASOLVER_H_

#include <musolver.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"

namespace at::musa::solver {

#define MUSASOLVER_GETRF_ARGTYPES(Dtype) \
  mublasHandle_t handle, int m, int n, Dtype *dA, int ldda, int *ipiv, int *info

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

} // namespace at::musa::solver
#endif // TORCH_MUSA_CSRC_ATEN_LINALG_MUSASOLVER_H_
