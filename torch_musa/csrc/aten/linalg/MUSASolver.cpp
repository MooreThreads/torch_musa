#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <c10/macros/Export.h>

#include "torch_musa/csrc/aten/linalg/MUSASolver.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"

namespace at::musa::solver {

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 4030)
#define TORCH_MUSOLVER_CHECK TORCH_MUSABLAS_CHECK

#define musolverDnDgetrf_bufferSize(handle, m, n, dA, ldda, lwork) \
  musolverDgetrf_bufferSize((m), (n), true, reinterpret_cast<size_t*>((lwork)))
#define musolverDnDgetrf(handle, m, n, dA, ldda, dataPtr, ipiv, info) \
  musolverDgetrf(                                                     \
      (handle),                                                       \
      (m),                                                            \
      (n),                                                            \
      (dA),                                                           \
      (ldda),                                                         \
      (ipiv),                                                         \
      (info),                                                         \
      static_cast<void*>((dataPtr)))
#define musolverDnSgetrf_bufferSize(handle, m, n, dA, ldda, lwork) \
  musolverSgetrf_bufferSize((m), (n), true, reinterpret_cast<size_t*>((lwork)))
#define musolverDnSgetrf(handle, m, n, dA, ldda, dataPtr, ipiv, info) \
  musolverSgetrf(                                                     \
      (handle),                                                       \
      (m),                                                            \
      (n),                                                            \
      (dA),                                                           \
      (ldda),                                                         \
      (ipiv),                                                         \
      (info),                                                         \
      static_cast<void*>((dataPtr)))
#define musolverDnZgetrf_bufferSize(handle, m, n, dA, ldda, lwork) \
  musolverZgetrf_bufferSize((m), (n), true, reinterpret_cast<size_t*>((lwork)))
#define musolverDnZgetrf(handle, m, n, dA, ldda, dataPtr, ipiv, info) \
  musolverZgetrf(                                                     \
      (handle),                                                       \
      (m),                                                            \
      (n),                                                            \
      (dA),                                                           \
      (ldda),                                                         \
      (ipiv),                                                         \
      (info),                                                         \
      static_cast<void*>((dataPtr)))
#define musolverDnCgetrf_bufferSize(handle, m, n, dA, ldda, lwork) \
  musolverCgetrf_bufferSize((m), (n), true, reinterpret_cast<size_t*>((lwork)))
#define musolverDnCgetrf(handle, m, n, dA, ldda, dataPtr, ipiv, info) \
  musolverCgetrf(                                                     \
      (handle),                                                       \
      (m),                                                            \
      (n),                                                            \
      (dA),                                                           \
      (ldda),                                                         \
      (ipiv),                                                         \
      (info),                                                         \
      static_cast<void*>((dataPtr)))

#define musolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo) \
  musolverSpotrs((handle), (uplo), (n), (nrhs), (A), (lda), (B), (ldb))
#define musolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo) \
  musolverDpotrs((handle), (uplo), (n), (nrhs), (A), (lda), (B), (ldb))
#define musolverDnCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo) \
  musolverCpotrs((handle), (uplo), (n), (nrhs), (A), (lda), (B), (ldb))
#define musolverDnZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo) \
  musolverZpotrs((handle), (uplo), (n), (nrhs), (A), (lda), (B), (ldb))

#define musolverDnDgetrs(                                  \
    handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info) \
  musolverDgetrs(                                          \
      (handle),                                            \
      (trans),                                             \
      (n),                                                 \
      (nrhs),                                              \
      (dA),                                                \
      (lda),                                               \
      (ipiv),                                              \
      (ret),                                               \
      (ldb),                                               \
      (info))
#define musolverDnSgetrs(                                  \
    handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info) \
  musolverSgetrs(                                          \
      (handle),                                            \
      (trans),                                             \
      (n),                                                 \
      (nrhs),                                              \
      (dA),                                                \
      (lda),                                               \
      (ipiv),                                              \
      (ret),                                               \
      (ldb),                                               \
      (info))
#define musolverDnZgetrs(                                  \
    handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info) \
  musolverZgetrs(                                          \
      (handle),                                            \
      (trans),                                             \
      (n),                                                 \
      (nrhs),                                              \
      (dA),                                                \
      (lda),                                               \
      (ipiv),                                              \
      (ret),                                               \
      (ldb),                                               \
      (info))
#define musolverDnCgetrs(                                  \
    handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info) \
  musolverCgetrs(                                          \
      (handle),                                            \
      (trans),                                             \
      (n),                                                 \
      (nrhs),                                              \
      (dA),                                                \
      (lda),                                               \
      (ipiv),                                              \
      (ret),                                               \
      (ldb),                                               \
      (info))

#define musolverDnSpotrfBatched(handle, uplo, n, A, lda, info, batchSize) \
  musolverSpotrfBatched((handle), (uplo), (n), (A), (lda), (info), (batchSize))
#define musolverDnDpotrfBatched(handle, uplo, n, A, lda, info, batchSize) \
  musolverDpotrfBatched((handle), (uplo), (n), (A), (lda), (info), (batchSize))
#define musolverDnCpotrfBatched(handle, uplo, n, A, lda, info, batchSize) \
  musolverCpotrfBatched((handle), (uplo), (n), (A), (lda), (info), (batchSize))
#define musolverDnZpotrfBatched(handle, uplo, n, A, lda, info, batchSize) \
  musolverZpotrfBatched((handle), (uplo), (n), (A), (lda), (info), (batchSize))
#endif

template <>
void getrf<double>(
    mublasHandle_t handle,
    int m,
    int n,
    double* dA,
    int ldda,
    int* ipiv,
    int* info) {
  int lwork;
  TORCH_MUSOLVER_CHECK(
      musolverDnDgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(double) * lwork);
  TORCH_MUSOLVER_CHECK(musolverDnDgetrf(
      handle, m, n, dA, ldda, static_cast<double*>(dataPtr.get()), ipiv, info));
}

template <>
void getrf<float>(
    mublasHandle_t handle,
    int m,
    int n,
    float* dA,
    int ldda,
    int* ipiv,
    int* info) {
  int lwork;
  TORCH_MUSOLVER_CHECK(
      musolverDnSgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(float) * lwork);
  TORCH_MUSOLVER_CHECK(musolverDnSgetrf(
      handle, m, n, dA, ldda, static_cast<float*>(dataPtr.get()), ipiv, info));
}

template <>
void getrf<c10::complex<double>>(
    mublasHandle_t handle,
    int m,
    int n,
    c10::complex<double>* dA,
    int ldda,
    int* ipiv,
    int* info) {
  int lwork;
  TORCH_MUSOLVER_CHECK(musolverDnZgetrf_bufferSize(
      handle, m, n, reinterpret_cast<muDoubleComplex*>(dA), ldda, &lwork));
  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(muDoubleComplex) * lwork);
  TORCH_MUSOLVER_CHECK(musolverDnZgetrf(
      handle,
      m,
      n,
      reinterpret_cast<muDoubleComplex*>(dA),
      ldda,
      reinterpret_cast<muDoubleComplex*>(dataPtr.get()),
      ipiv,
      info));
}

template <>
void getrf<c10::complex<float>>(
    mublasHandle_t handle,
    int m,
    int n,
    c10::complex<float>* dA,
    int ldda,
    int* ipiv,
    int* info) {
  int lwork;
  TORCH_MUSOLVER_CHECK(musolverDnCgetrf_bufferSize(
      handle, m, n, reinterpret_cast<muComplex*>(dA), ldda, &lwork));
  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(muComplex) * lwork);
  TORCH_MUSOLVER_CHECK(musolverDnCgetrf(
      handle,
      m,
      n,
      reinterpret_cast<muComplex*>(dA),
      ldda,
      reinterpret_cast<muComplex*>(dataPtr.get()),
      ipiv,
      info));
}

template <>
void potrs<float>(
    mublasHandle_t handle,
    mublasFillMode_t uplo,
    int n,
    int nrhs,
    float* A,
    int lda,
    float* B,
    int ldb,
    int* devInfo) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo));
}

template <>
void potrs<double>(
    mublasHandle_t handle,
    mublasFillMode_t uplo,
    int n,
    int nrhs,
    double* A,
    int lda,
    double* B,
    int ldb,
    int* devInfo) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo));
}

template <>
void potrs<c10::complex<float>>(
    mublasHandle_t handle,
    mublasFillMode_t uplo,
    int n,
    int nrhs,
    c10::complex<float>* A,
    int lda,
    c10::complex<float>* B,
    int ldb,
    int* devInfo) {
  TORCH_MUSOLVER_CHECK(musolverDnCpotrs(
      handle,
      uplo,
      n,
      nrhs,
      reinterpret_cast<muComplex*>(A),
      lda,
      reinterpret_cast<muComplex*>(B),
      ldb,
      devInfo));
}

template <>
void potrs<c10::complex<double>>(
    mublasHandle_t handle,
    mublasFillMode_t uplo,
    int n,
    int nrhs,
    c10::complex<double>* A,
    int lda,
    c10::complex<double>* B,
    int ldb,
    int* devInfo) {
  TORCH_MUSOLVER_CHECK(musolverDnZpotrs(
      handle,
      uplo,
      n,
      nrhs,
      reinterpret_cast<muDoubleComplex*>(A),
      lda,
      reinterpret_cast<muDoubleComplex*>(B),
      ldb,
      devInfo));
}

template <>
void getrs<double>(
    mublasHandle_t handle,
    int n,
    int nrhs,
    double* dA,
    int lda,
    int* ipiv,
    double* ret,
    int ldb,
    int* info,
    mublasOperation_t trans) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDgetrs(handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info));
}

template <>
void getrs<float>(
    mublasHandle_t handle,
    int n,
    int nrhs,
    float* dA,
    int lda,
    int* ipiv,
    float* ret,
    int ldb,
    int* info,
    mublasOperation_t trans) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSgetrs(handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info));
}

template <>
void getrs<c10::complex<double>>(
    mublasHandle_t handle,
    int n,
    int nrhs,
    c10::complex<double>* dA,
    int lda,
    int* ipiv,
    c10::complex<double>* ret,
    int ldb,
    int* info,
    mublasOperation_t trans) {
  TORCH_MUSOLVER_CHECK(musolverDnZgetrs(
      handle,
      trans,
      n,
      nrhs,
      reinterpret_cast<muDoubleComplex*>(dA),
      lda,
      ipiv,
      reinterpret_cast<muDoubleComplex*>(ret),
      ldb,
      info));
}

template <>
void getrs<c10::complex<float>>(
    mublasHandle_t handle,
    int n,
    int nrhs,
    c10::complex<float>* dA,
    int lda,
    int* ipiv,
    c10::complex<float>* ret,
    int ldb,
    int* info,
    mublasOperation_t trans) {
  TORCH_MUSOLVER_CHECK(musolverDnCgetrs(
      handle,
      trans,
      n,
      nrhs,
      reinterpret_cast<muComplex*>(dA),
      lda,
      ipiv,
      reinterpret_cast<muComplex*>(ret),
      ldb,
      info));
}

template <>
void potrfBatched<float>(
    mublasHandle_t handle,
    mublasFillMode_t uplo,
    int n,
    float** A,
    int lda,
    int* info,
    int batchSize) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSpotrfBatched(handle, uplo, n, A, lda, info, batchSize));
}

template <>
void potrfBatched<double>(
    mublasHandle_t handle,
    mublasFillMode_t uplo,
    int n,
    double** A,
    int lda,
    int* info,
    int batchSize) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDpotrfBatched(handle, uplo, n, A, lda, info, batchSize));
}

template <>
void potrfBatched<c10::complex<float>>(
    mublasHandle_t handle,
    mublasFillMode_t uplo,
    int n,
    c10::complex<float>** A,
    int lda,
    int* info,
    int batchSize) {
  TORCH_MUSOLVER_CHECK(musolverDnCpotrfBatched(
      handle, uplo, n, reinterpret_cast<muComplex**>(A), lda, info, batchSize));
}

template <>
void potrfBatched<c10::complex<double>>(
    mublasHandle_t handle,
    mublasFillMode_t uplo,
    int n,
    c10::complex<double>** A,
    int lda,
    int* info,
    int batchSize) {
  TORCH_MUSOLVER_CHECK(musolverDnZpotrfBatched(
      handle,
      uplo,
      n,
      reinterpret_cast<muDoubleComplex**>(A),
      lda,
      info,
      batchSize));
}
} // namespace at::musa::solver