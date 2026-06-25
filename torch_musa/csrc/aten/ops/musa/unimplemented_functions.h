#ifndef TORCH_MUSA_CSRC_ATEN_OPS_MUSA_UNIMPLEMENTED_FUNCTIONS_H
#define TORCH_MUSA_CSRC_ATEN_OPS_MUSA_UNIMPLEMENTED_FUNCTIONS_H

#include <musa_runtime_api.h>
#include <musparse.h>
#include <memory>

#include "unimplemented_functions_sparse.h"

typedef struct _musparse_mat_info bsrsv2Info;
typedef struct _musparse_mat_info bsrsm2Info;

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
    size_t* pBufferSizeInBytes);

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
    void* pBuffer);

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
    void* pBuffer);

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
    size_t* pBufferSizeInBytes);

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
    void* pBuffer);

using musparseMatDescr = std::remove_pointer_t<musparseMatDescr_t>;
using musparseDnMatDescr = std::remove_pointer_t<musparseDnMatDescr_t>;
using musparseDnVecDescr = std::remove_pointer_t<musparseDnVecDescr_t>;
using musparseSpMatDescr = std::remove_pointer_t<musparseSpMatDescr_t>;
using musparseSpMatDescr = std::remove_pointer_t<musparseSpMatDescr_t>;
using musparseSpSVDescr = std::remove_pointer_t<musparseSpSVDescr_t>;

namespace c10 {
namespace musa {

__host__ musaError_t musaStreamGetCaptureInfo(
    musaStream_t stream,
    enum musaStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out);

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 4020)
__host__ musaError_t
musaGraphDebugDotPrint(musaGraph_t graph, const char* path, unsigned int flags);
#endif

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 5010)
__host__ musaError_t
musaMemPoolTrimTo(musaMemPool_t memPool, size_t minBytesToKeep);

__host__ musaError_t
musaMallocAsync(void** devPtr, size_t size, musaStream_t hStream);

__host__ musaError_t musaFreeAsync(void* devPtr, musaStream_t hStream);

__host__ musaError_t
musaDeviceGetDefaultMemPool(musaMemPool_t* memPool, int device);

__host__ musaError_t
musaThreadExchangeStreamCaptureMode(enum musaStreamCaptureMode* mode);

__host__ musaError_t musaMemPoolGetAttribute(
    musaMemPool_t memPool,
    enum musaMemPoolAttr attr,
    void* value);

__host__ musaError_t musaMemPoolSetAttribute(
    musaMemPool_t memPool,
    enum musaMemPoolAttr attr,
    void* value);

#endif
} // namespace musa
} // namespace c10

namespace at {
class Tensor;

namespace native::sparse {
using SparseTensor = at::Tensor;

} // namespace native::sparse
} // namespace at

#endif // TORCH_MUSA_CSRC_ATEN_OPS_MUSA_UNIMPLEMENTED_FUNCTIONS_H
