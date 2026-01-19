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
void potrsBatched<float>(
    mublasHandle_t handle,
    mublasFillMode_t uplo,
    int n,
    int nrhs,
    float* Aarray[],
    int lda,
    float* Barray[],
    int ldb,
    int* info,
    int batchSize) {
  TORCH_MUSOLVER_CHECK(musolverDnSpotrsBatched(
      handle, uplo, n, nrhs, Aarray, lda, Barray, ldb, info, batchSize));
}

template <>
void potrsBatched<double>(
    mublasHandle_t handle,
    mublasFillMode_t uplo,
    int n,
    int nrhs,
    double* Aarray[],
    int lda,
    double* Barray[],
    int ldb,
    int* info,
    int batchSize) {
  TORCH_MUSOLVER_CHECK(musolverDnDpotrsBatched(
      handle, uplo, n, nrhs, Aarray, lda, Barray, ldb, info, batchSize));
}

template <>
void potrsBatched<c10::complex<float>>(
    mublasHandle_t handle,
    mublasFillMode_t uplo,
    int n,
    int nrhs,
    c10::complex<float>* Aarray[],
    int lda,
    c10::complex<float>* Barray[],
    int ldb,
    int* info,
    int batchSize) {
  TORCH_MUSOLVER_CHECK(musolverDnCpotrsBatched(
      handle,
      uplo,
      n,
      nrhs,
      reinterpret_cast<muComplex**>(Aarray),
      lda,
      reinterpret_cast<muComplex**>(Barray),
      ldb,
      info,
      batchSize));
}

template <>
void potrsBatched<c10::complex<double>>(
    mublasHandle_t handle,
    mublasFillMode_t uplo,
    int n,
    int nrhs,
    c10::complex<double>* Aarray[],
    int lda,
    c10::complex<double>* Barray[],
    int ldb,
    int* info,
    int batchSize) {
  TORCH_MUSOLVER_CHECK(musolverDnZpotrsBatched(
      handle,
      uplo,
      n,
      nrhs,
      reinterpret_cast<muDoubleComplex**>(Aarray),
      lda,
      reinterpret_cast<muDoubleComplex**>(Barray),
      ldb,
      info,
      batchSize));
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

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 5010)
template <>
void geqrf_bufferSize<float>(MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(musolverDnSgeqrf_bufferSize(m, n, lwork));
}

template <>
void geqrf_bufferSize<double>(MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(musolverDnDgeqrf_bufferSize(m, n, lwork));
}

template <>
void geqrf_bufferSize<c10::complex<float>>(
    MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnCgeqrf_bufferSize(m, n, lwork));
}

template <>
void geqrf_bufferSize<c10::complex<double>>(
    MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZgeqrf_bufferSize(m, n, lwork));
}

template <>
void geqrf<float>(MUSASOLVER_GEQRF_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(musolverDnSgeqrf(handle, m, n, dA, lda, ipiv, buffer));
}

template <>
void geqrf<double>(MUSASOLVER_GEQRF_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(musolverDnDgeqrf(handle, m, n, dA, lda, ipiv, buffer));
}

template <>
void geqrf<c10::complex<float>>(
    MUSASOLVER_GEQRF_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnCgeqrf(
      handle,
      m,
      n,
      reinterpret_cast<muComplex*>(dA),
      lda,
      reinterpret_cast<muComplex*>(ipiv),
      buffer));
}

template <>
void geqrf<c10::complex<double>>(
    MUSASOLVER_GEQRF_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZgeqrf(
      handle,
      m,
      n,
      reinterpret_cast<muDoubleComplex*>(dA),
      lda,
      reinterpret_cast<muDoubleComplex*>(ipiv),
      buffer));
}

template <>
void orgqr_buffersize<float>(int m, int n, int k, size_t* lwork) {
  TORCH_MUSOLVER_CHECK(musolverDnSorgqr_bufferSize(m, n, k, lwork));
}

template <>
void orgqr_buffersize<double>(int m, int n, int k, size_t* lwork) {
  TORCH_MUSOLVER_CHECK(musolverDnDorgqr_bufferSize(m, n, k, lwork));
}

template <>
void orgqr_buffersize<c10::complex<float>>(int m, int n, int k, size_t* lwork) {
  TORCH_MUSOLVER_CHECK(musolverDnCungqr_bufferSize(m, n, k, lwork));
}

template <>
void orgqr_buffersize<c10::complex<double>>(
    int m,
    int n,
    int k,
    size_t* lwork) {
  TORCH_MUSOLVER_CHECK(musolverDnZungqr_bufferSize(m, n, k, lwork));
}

template <>
void orgqr<float>(MUSASOLVER_ORGQR_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(musolverDnSorgqr(handle, m, n, k, A, lda, ipiv, buffer));
}

template <>
void orgqr<double>(MUSASOLVER_ORGQR_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(musolverDnDorgqr(handle, m, n, k, A, lda, ipiv, buffer));
}

template <>
void orgqr<c10::complex<float>>(
    MUSASOLVER_ORGQR_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnCungqr(
      handle,
      m,
      n,
      k,
      reinterpret_cast<muComplex*>(A),
      lda,
      reinterpret_cast<muComplex*>(ipiv),
      buffer));
}

template <>
void orgqr<c10::complex<double>>(
    MUSASOLVER_ORGQR_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZungqr(
      handle,
      m,
      n,
      k,
      reinterpret_cast<muDoubleComplex*>(A),
      lda,
      reinterpret_cast<muDoubleComplex*>(ipiv),
      buffer));
}

template <>
void ormqr<float>(MUSASOLVER_ORMQR_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSormqr(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc));
}

template <>
void ormqr<double>(MUSASOLVER_ORMQR_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDormqr(handle, side, trans, m, n, k, A, lda, ipiv, C, ldc));
}

template <>
void ormqr<c10::complex<float>>(
    MUSASOLVER_ORMQR_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnCunmqr(
      handle,
      side,
      trans,
      m,
      n,
      k,
      reinterpret_cast<muComplex*>(A),
      lda,
      reinterpret_cast<muComplex*>(ipiv),
      reinterpret_cast<muComplex*>(C),
      ldc));
}

template <>
void ormqr<c10::complex<double>>(
    MUSASOLVER_ORMQR_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZunmqr(
      handle,
      side,
      trans,
      m,
      n,
      k,
      reinterpret_cast<muDoubleComplex*>(A),
      lda,
      reinterpret_cast<muDoubleComplex*>(ipiv),
      reinterpret_cast<muDoubleComplex*>(C),
      ldc));
}

#else
template <>
void geqrf_bufferSize<float>(MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork));
}

template <>
void geqrf_bufferSize<double>(MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork));
}

template <>
void geqrf_bufferSize<c10::complex<float>>(
    MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnCgeqrf_bufferSize(
      handle, m, n, reinterpret_cast<muComplex*>(A), lda, lwork));
}

template <>
void geqrf_bufferSize<c10::complex<double>>(
    MUSASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZgeqrf_bufferSize(
      handle, m, n, reinterpret_cast<muDoubleComplex*>(A), lda, lwork));
}

template <>
void geqrf<float>(MUSASOLVER_GEQRF_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo));
}

template <>
void geqrf<double>(MUSASOLVER_GEQRF_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo));
}

template <>
void geqrf<c10::complex<float>>(
    MUSASOLVER_GEQRF_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnCgeqrf(
      handle,
      m,
      n,
      reinterpret_cast<muComplex*>(A),
      lda,
      reinterpret_cast<muComplex*>(tau),
      reinterpret_cast<muComplex*>(work),
      lwork,
      devInfo));
}

template <>
void geqrf<c10::complex<double>>(
    MUSASOLVER_GEQRF_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZgeqrf(
      handle,
      m,
      n,
      reinterpret_cast<muDoubleComplex*>(A),
      lda,
      reinterpret_cast<muDoubleComplex*>(tau),
      reinterpret_cast<muDoubleComplex*>(work),
      lwork,
      devInfo));
}

template <>
void orgqr_buffersize<float>(MUSASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork));
}

template <>
void orgqr_buffersize<double>(MUSASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork));
}

template <>
void orgqr_buffersize<c10::complex<float>>(
    MUSASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnCungqr_bufferSize(
      handle,
      m,
      n,
      k,
      reinterpret_cast<const muComplex*>(A),
      lda,
      reinterpret_cast<const muComplex*>(tau),
      lwork));
}

template <>
void orgqr_buffersize<c10::complex<double>>(
    MUSASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZungqr_bufferSize(
      handle,
      m,
      n,
      k,
      reinterpret_cast<const muDoubleComplex*>(A),
      lda,
      reinterpret_cast<const muDoubleComplex*>(tau),
      lwork));
}

template <>
void orgqr<float>(MUSASOLVER_ORGQR_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo));
}

template <>
void orgqr<double>(MUSASOLVER_ORGQR_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(
      musolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo));
}

template <>
void orgqr<c10::complex<float>>(
    MUSASOLVER_ORGQR_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnCungqr(
      handle,
      m,
      n,
      k,
      reinterpret_cast<muComplex*>(A),
      lda,
      reinterpret_cast<muComplex*>(tau),
      reinterpret_cast<muComplex*>(work),
      lwork,
      devInfo));
}

template <>
void orgqr<c10::complex<double>>(
    MUSASOLVER_ORGQR_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZungqr(
      handle,
      m,
      n,
      k,
      reinterpret_cast<muDoubleComplex*>(A),
      lda,
      reinterpret_cast<muDoubleComplex*>(tau),
      reinterpret_cast<muDoubleComplex*>(work),
      lwork,
      devInfo));
}

template <>
void ormqr_buffersize<float>(MUSASOLVER_ORMQR_BUFFERSIZE_ARGTYPES_REAL(float)) {
  TORCH_MUSOLVER_CHECK(musolverDnSormqr_bufferSize(
      handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork));
}

template <>
void ormqr_buffersize<double>(
    MUSASOLVER_ORMQR_BUFFERSIZE_ARGTYPES_REAL(double)) {
  TORCH_MUSOLVER_CHECK(musolverDnDormqr_bufferSize(
      handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork));
}

template <>
void ormqr<float>(MUSASOLVER_ORMQR_ARGTYPES(float)) {
  TORCH_MUSOLVER_CHECK(musolverDnSormqr(
      handle,
      side,
      trans,
      m,
      n,
      k,
      A,
      lda,
      ipiv,
      C,
      ldc,
      buffer,
      lwork,
      devInfo));
}

template <>
void ormqr<double>(MUSASOLVER_ORMQR_ARGTYPES(double)) {
  TORCH_MUSOLVER_CHECK(musolverDnDormqr(
      handle,
      side,
      trans,
      m,
      n,
      k,
      A,
      lda,
      ipiv,
      C,
      ldc,
      buffer,
      lwork,
      devInfo));
}

template <>
void ormqr<c10::complex<float>>(
    MUSASOLVER_ORMQR_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSOLVER_CHECK(musolverDnCunmqr(
      handle,
      side,
      trans,
      m,
      n,
      k,
      reinterpret_cast<muComplex*>(A),
      lda,
      reinterpret_cast<muComplex*>(ipiv),
      reinterpret_cast<muComplex*>(C),
      ldc));
}

template <>
void ormqr<c10::complex<double>>(
    MUSASOLVER_ORMQR_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSOLVER_CHECK(musolverDnZunmqr(
      handle,
      side,
      trans,
      m,
      n,
      k,
      reinterpret_cast<muDoubleComplex*>(A),
      lda,
      reinterpret_cast<muDoubleComplex*>(ipiv),
      reinterpret_cast<muDoubleComplex*>(C),
      ldc));
}

#endif

} // namespace at::musa::solver
