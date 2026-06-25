#include "torch_musa/csrc/aten/ops/musa/unimplemented_functions.h"
#include <c10/util/Exception.h>
#include <musa_runtime_api.h>

#define MUSA_UNIMPLEMENTED_STATUS_FUNC()                                      \
  C10_THROW_ERROR(                                                            \
      NotImplementedError, std::string(__func__) + " is not supported now!"); \
  return musparseStatus_t{};

musparseStatus_t musparseDcsrgemm2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const double* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const double* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    csrgemm2Info_t info,
    size_t* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseXcsrgemm2Nnz(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const musparseMatDescr_t descrA,
    int nnzA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrD,
    int nnzD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    const musparseMatDescr_t descrC,
    int* csrSortedRowPtrC,
    int* nnzTotalDevHostPtr,
    const csrgemm2Info_t info,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseDcsrgemm2(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const double* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const double* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const double* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const double* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const double* csrSortedValD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    const musparseMatDescr_t descrC,
    double* csrSortedValC,
    const int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    const csrgemm2Info_t info,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseScsrgemm2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const float* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const float* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    csrgemm2Info_t info,
    size_t* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseScsrgemm2(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const float* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const float* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const float* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const float* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const float* csrSortedValD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    const musparseMatDescr_t descrC,
    float* csrSortedValC,
    const int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    const csrgemm2Info_t info,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

namespace c10 {
namespace musa {

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 4020)
__host__ musaError_t musaGraphDebugDotPrint(
    musaGraph_t graph,
    const char* path,
    unsigned int flags) {
  C10_THROW_ERROR(
      NotImplementedError, "musaGraphDebugDotPrint is not supported now!");
  return musaErrorNotSupported;
}
#endif

__host__ musaError_t musaStreamGetCaptureInfo(
    musaStream_t stream,
    enum musaStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out) {
  C10_THROW_ERROR(
      NotImplementedError, "musaStreamGetCaptureInfo is not supported now!");
  return musaErrorNotSupported;
}

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 5010)
__host__ musaError_t
musaThreadExchangeStreamCaptureMode(enum musaStreamCaptureMode* mode) {
  C10_THROW_ERROR(
      NotImplementedError,
      "musaThreadExchangeStreamCaptureMode is not supported now!");
  return musaErrorNotSupported;
}

__host__ musaError_t
musaMemPoolTrimTo(musaMemPool_t memPool, size_t minBytesToKeep) {
  C10_THROW_ERROR(
      NotImplementedError, "musaMemPoolTrimTo is not supported now!");
  return musaErrorNotSupported;
}

__host__ musaError_t
musaMallocAsync(void** devPtr, size_t size, musaStream_t hStream) {
  C10_THROW_ERROR(NotImplementedError, "musaMallocAsync is not supported now!");
  return musaErrorNotSupported;
}

__host__ musaError_t musaFreeAsync(void* devPtr, musaStream_t hStream) {
  C10_THROW_ERROR(NotImplementedError, "musaFreeAsync is not supported now!");
  return musaErrorNotSupported;
}

__host__ musaError_t
musaDeviceGetDefaultMemPool(musaMemPool_t* memPool, int device) {
  C10_THROW_ERROR(
      NotImplementedError, "musaDeviceGetDefaultMemPool is not supported now!");
  return musaErrorNotSupported;
}

__host__ musaError_t musaMemPoolGetAttribute(
    musaMemPool_t memPool,
    enum musaMemPoolAttr attr,
    void* value) {
  C10_THROW_ERROR(
      NotImplementedError, "musaMemPoolGetAttribute is not supported now!");
  return musaErrorNotSupported;
}

__host__ musaError_t musaMemPoolSetAttribute(
    musaMemPool_t memPool,
    enum musaMemPoolAttr attr,
    void* value) {
  C10_THROW_ERROR(
      NotImplementedError, "musaMemPoolSetAttribute is not supported now!");
  return musaErrorNotSupported;
}
#endif

} // namespace musa
} // namespace c10
