#include "torch_musa/csrc/aten/ops/musa/unimplemented_functions_sparse.h"
#include <c10/util/Exception.h>
#include <musa_runtime_api.h>

#if defined(MUSPARSE_VERSION) && (MUSPARSE_VERSION < 12000)
#define MUSA_UNIMPLEMENTED_STATUS_FUNC()                                      \
  C10_THROW_ERROR(                                                            \
      NotImplementedError, std::string(__func__) + " is not supported now!"); \
  return musparseStatus_t{};

const char* musparseGetErrorString(musparseStatus_t status) {
  switch (status) {
    case MUSPARSE_STATUS_SUCCESS:
      return "success";

    case MUSPARSE_STATUS_NOT_INITIALIZED:
      return "library not initialized";

    case MUSPARSE_STATUS_ALLOC_FAILED:
      return "resource allocation failed";

    case MUSPARSE_STATUS_INVALID_VALUE:
      return "an invalid numeric value was used as an argument";

    case MUSPARSE_STATUS_ARCH_MISMATCH:
      return "an absent device architectural feature is required";

    case MUSPARSE_STATUS_MAPPING_ERROR:
      return "an access to GPU memory space failed";

    case MUSPARSE_STATUS_EXECUTION_FAILED:
      return "the GPU program failed to execute";

    case MUSPARSE_STATUS_INTERNAL_ERROR:
      return "an internal operation failed";

    case MUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "the matrix type is not supported by this function";

    case MUSPARSE_STATUS_ZERO_PIVOT:
      return "an entry of the matrix is either structural zero or numerical zero (singular block)";

    default:
      return "unknown error";
  }
}

musparseStatus_t musparseDestroyDnMat(musparseConstDnMatDescr_t descr) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseDestroyDnVec(musparseConstDnVecDescr_t descr) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

namespace at {
namespace sparse {
musparseStatus_t musparseSpGEMM_destroyDescr(musparseSpGEMMDescr_t descr) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseSpGEMM_createDescr(musparseSpGEMMDescr_t* descr) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseSpGEMM_workEstimation(
    musparseHandle_t handle,
    musparseOperation_t opA,
    musparseOperation_t opB,
    const void* alpha,
    musparseSpMatDescr_t matA,
    musparseSpMatDescr_t matB,
    const void* beta,
    musparseSpMatDescr_t matC,
    musaDataType computeType,
    musparseSpGEMMAlg_t alg,
    musparseSpGEMMDescr_t spgemmDescr,
    size_t* bufferSize1,
    void* externalBuffer1) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseSpGEMM_compute(
    musparseHandle_t handle,
    musparseOperation_t opA,
    musparseOperation_t opB,
    const void* alpha,
    musparseSpMatDescr_t matA,
    musparseSpMatDescr_t matB,
    const void* beta,
    musparseSpMatDescr_t matC,
    musaDataType computeType,
    musparseSpGEMMAlg_t alg,
    musparseSpGEMMDescr_t spgemmDescr,
    size_t* bufferSize2,
    void* externalBuffer2) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseSpGEMM_copy(
    musparseHandle_t handle,
    musparseOperation_t opA,
    musparseOperation_t opB,
    const void* alpha,
    musparseSpMatDescr_t matA,
    musparseSpMatDescr_t matB,
    const void* beta,
    musparseSpMatDescr_t matC,
    musaDataType computeType,
    musparseSpGEMMAlg_t alg,
    musparseSpGEMMDescr_t spgemmDescr) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

} // namespace sparse
namespace musa {
namespace sparse {
musparseStatus_t musparseCreateBsrsv2Info(bsrsv2Info_t* info) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseDestroyBsrsv2Info(bsrsv2Info_t info) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseCreateBsrsm2Info(bsrsm2Info_t* info) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseDestroyBsrsm2Info(bsrsm2Info_t info) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseCcsrgeam2(
    musparseHandle_t handle,
    int m,
    int n,
    const muComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const muComplex* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const muComplex* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const muComplex* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    muComplex* csrSortedValC,
    int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseDcsrgeam2(
    musparseHandle_t handle,
    int m,
    int n,
    const double* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const double* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const double* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const double* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    double* csrSortedValC,
    int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseScsrgeam2(
    musparseHandle_t handle,
    int m,
    int n,
    const float* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const float* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const float* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const float* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    float* csrSortedValC,
    int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseZcsrgeam2(
    musparseHandle_t handle,
    int m,
    int n,
    const muDoubleComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const muDoubleComplex* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const muDoubleComplex* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const muDoubleComplex* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    muDoubleComplex* csrSortedValC,
    int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseCcsrgeam2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    const muComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const muComplex* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const muComplex* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const muComplex* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    const muComplex* csrSortedValC,
    const int* csrSortedRowPtrC,
    const int* csrSortedColIndC,
    size_t* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseDcsrgeam2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    const double* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const double* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const double* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const double* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    const double* csrSortedValC,
    const int* csrSortedRowPtrC,
    const int* csrSortedColIndC,
    size_t* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseScsrgeam2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    const float* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const float* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const float* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const float* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    const float* csrSortedValC,
    const int* csrSortedRowPtrC,
    const int* csrSortedColIndC,
    size_t* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseZcsrgeam2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    const muDoubleComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const muDoubleComplex* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const muDoubleComplex* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const muDoubleComplex* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    const muDoubleComplex* csrSortedValC,
    const int* csrSortedRowPtrC,
    const int* csrSortedColIndC,
    size_t* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseCbsrsv2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    muComplex* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    int* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseDbsrsv2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    double* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    int* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseSbsrsv2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    float* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    int* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseZbsrsv2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    muDoubleComplex* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    int* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseDbsrsv2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    const double* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseCbsrsv2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    const muComplex* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseSbsrsv2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    const float* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseZbsrsv2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    const muDoubleComplex* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseCbsrsm2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const muComplex* alpha,
    const musparseMatDescr_t descrA,
    const muComplex* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    const muComplex* B,
    int ldb,
    muComplex* X,
    int ldx,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseDbsrsm2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const double* alpha,
    const musparseMatDescr_t descrA,
    const double* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    const double* B,
    int ldb,
    double* X,
    int ldx,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseSbsrsm2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const float* alpha,
    const musparseMatDescr_t descrA,
    const float* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    const float* B,
    int ldb,
    float* X,
    int ldx,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseZbsrsm2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const muDoubleComplex* alpha,
    const musparseMatDescr_t descrA,
    const muDoubleComplex* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    const muDoubleComplex* B,
    int ldb,
    muDoubleComplex* X,
    int ldx,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseSbsrsv2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const float* alpha,
    const musparseMatDescr_t descrA,
    const float* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    const float* f,
    float* x,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseCbsrsv2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const muComplex* alpha,
    const musparseMatDescr_t descrA,
    const muComplex* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    const muComplex* f,
    muComplex* x,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseDbsrsv2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const double* alpha,
    const musparseMatDescr_t descrA,
    const double* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    const double* f,
    double* x,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseZbsrsv2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const muDoubleComplex* alpha,
    const musparseMatDescr_t descrA,
    const muDoubleComplex* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    const muDoubleComplex* f,
    muDoubleComplex* x,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseCbsrsm2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    muComplex* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    int* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseDbsrsm2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    double* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    int* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseSbsrsm2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    float* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    int* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseZbsrsm2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    muDoubleComplex* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    int* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseCbsrsm2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    const muComplex* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseDbsrsm2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    const double* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseSbsrsm2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    const float* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseZbsrsm2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    const muDoubleComplex* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseXcsrgeam2Nnz(
    musparseHandle_t handle,
    int m,
    int n,
    const musparseMatDescr_t descrA,
    int nnzA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    int* csrSortedRowPtrC,
    int* nnzTotalDevHostPtr,
    void* workspace) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
} // namespace sparse
} // namespace musa

namespace native {

musparseStatus_t musparseSpMM_bufferSize(
    musparseHandle_t handle,
    musparseOperation_t opA,
    musparseOperation_t opB,
    const void* alpha,
    musparseSpMatDescr_t matA,
    musparseDnMatDescr_t matB,
    const void* beta,
    musparseDnMatDescr_t matC,
    musaDataType computeType,
    musparseSpMMAlg_t alg,
    size_t* bufferSize) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseSpMM(
    musparseHandle_t handle,
    musparseOperation_t opA,
    musparseOperation_t opB,
    const void* alpha,
    musparseSpMatDescr_t matA,
    musparseDnMatDescr_t matB,
    const void* beta,
    musparseDnMatDescr_t matC,
    musaDataType computeType,
    musparseSpMMAlg_t alg,
    void* externalBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseCcsrgemm2(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const muComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const muComplex* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const muComplex* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const muComplex* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const muComplex* csrSortedValD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    const musparseMatDescr_t descrC,
    muComplex* csrSortedValC,
    const int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    const csrgemm2Info_t info,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseZcsrgemm2(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const muDoubleComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const muDoubleComplex* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const muDoubleComplex* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const muDoubleComplex* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const muDoubleComplex* csrSortedValD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    const musparseMatDescr_t descrC,
    muDoubleComplex* csrSortedValC,
    const int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    const csrgemm2Info_t info,
    void* pBuffer) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseCcsrgemm2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const muComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const muComplex* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    csrgemm2Info_t info,
    size_t* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

musparseStatus_t musparseZcsrgemm2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const muDoubleComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const muDoubleComplex* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    csrgemm2Info_t info,
    size_t* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

namespace sparse {
namespace musa {
musparseStatus_t musparseXcsrsort_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const int* csrRowPtrA,
    const int* csrColIndA,
    size_t* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseXcoosort_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const int* cooRowsA,
    const int* cooColsA,
    size_t* pBufferSizeInBytes) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
} // namespace musa
namespace impl::musa {
musparseStatus_t musparseXbsrsv2_zeroPivot(
    musparseHandle_t handle,
    at::musa::sparse::bsrsv2Info_t info,
    int* position) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}
musparseStatus_t musparseXbsrsm2_zeroPivot(
    musparseHandle_t handle,
    at::musa::sparse::bsrsm2Info_t info,
    int* position) {
  MUSA_UNIMPLEMENTED_STATUS_FUNC();
}

} // namespace impl::musa
} // namespace sparse

} // namespace native
} // namespace at

#endif
