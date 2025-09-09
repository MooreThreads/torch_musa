#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <c10/macros/Export.h>

#include "torch_musa/csrc/aten/linalg/MUSASolver.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"

namespace at::musa::solver {

template <>
void getrf<double>(
    mublasHandle_t handle,
    int m,
    int n,
    double* dA,
    int ldda,
    int* ipiv,
    int* info) {
  size_t lwork;
  TORCH_MUSABLAS_CHECK(musolverDgetrf_bufferSize(m, n, true, &lwork));
  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(double) * lwork);
  TORCH_MUSABLAS_CHECK(musolverDgetrf(
      handle, m, n, dA, ldda, ipiv, info, static_cast<void*>(dataPtr.get())));
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
  size_t lwork;
  TORCH_MUSABLAS_CHECK(musolverSgetrf_bufferSize(m, n, true, &lwork));
  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(float) * lwork);
  TORCH_MUSABLAS_CHECK(musolverSgetrf(
      handle, m, n, dA, ldda, ipiv, info, static_cast<void*>(dataPtr.get())));
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
  size_t lwork;
  TORCH_MUSABLAS_CHECK(musolverZgetrf_bufferSize(m, n, true, &lwork));
  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(muDoubleComplex) * lwork);
  TORCH_MUSABLAS_CHECK(musolverZgetrf(
      handle,
      m,
      n,
      reinterpret_cast<muDoubleComplex*>(dA),
      ldda,
      ipiv,
      info,
      static_cast<void*>(dataPtr.get())));
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
  size_t lwork;
  TORCH_MUSABLAS_CHECK(musolverCgetrf_bufferSize(m, n, true, &lwork));
  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(muComplex) * lwork);
  TORCH_MUSABLAS_CHECK(musolverCgetrf(
      handle,
      m,
      n,
      reinterpret_cast<muComplex*>(dA),
      ldda,
      ipiv,
      info,
      static_cast<void*>(dataPtr.get())));
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
  TORCH_MUSABLAS_CHECK(musolverSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb));
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
  TORCH_MUSABLAS_CHECK(musolverDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb));
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
  TORCH_MUSABLAS_CHECK(musolverCpotrs(
      handle,
      uplo,
      n,
      nrhs,
      reinterpret_cast<muComplex*>(A),
      lda,
      reinterpret_cast<muComplex*>(B),
      ldb));
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
  TORCH_MUSABLAS_CHECK(musolverZpotrs(
      handle,
      uplo,
      n,
      nrhs,
      reinterpret_cast<muDoubleComplex*>(A),
      lda,
      reinterpret_cast<muDoubleComplex*>(B),
      ldb));
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
  TORCH_MUSABLAS_CHECK(
      musolverDgetrs(handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info));
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
  TORCH_MUSABLAS_CHECK(
      musolverSgetrs(handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info));
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
  TORCH_MUSABLAS_CHECK(musolverZgetrs(
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
  TORCH_MUSABLAS_CHECK(musolverCgetrs(
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
  TORCH_MUSABLAS_CHECK(
      musolverSpotrfBatched(handle, uplo, n, A, lda, info, batchSize));
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
  TORCH_MUSABLAS_CHECK(
      musolverDpotrfBatched(handle, uplo, n, A, lda, info, batchSize));
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
  TORCH_MUSABLAS_CHECK(musolverCpotrfBatched(
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
  TORCH_MUSABLAS_CHECK(musolverZpotrfBatched(
      handle,
      uplo,
      n,
      reinterpret_cast<muDoubleComplex**>(A),
      lda,
      info,
      batchSize));
}

} // namespace at::musa::solver