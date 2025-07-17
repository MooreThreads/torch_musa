#ifndef TORCH_MUSA_CSRC_ATEN_OPS_MUSA_UNIMPLEMENTED_FUNCTIONS_H
#define TORCH_MUSA_CSRC_ATEN_OPS_MUSA_UNIMPLEMENTED_FUNCTIONS_H

#include <musa_runtime_api.h>

namespace c10 {
namespace musa {

__host__ musaError_t
musaThreadExchangeStreamCaptureMode(enum musaStreamCaptureMode* mode);

__host__ musaError_t musaStreamGetCaptureInfo(
    musaStream_t stream,
    enum musaStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out);

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 4020)
__host__ musaError_t
musaGraphDebugDotPrint(musaGraph_t graph, const char* path, unsigned int flags);
#endif

__host__ musaError_t
musaMemPoolTrimTo(musaMemPool_t memPool, size_t minBytesToKeep);

__host__ musaError_t
musaMallocAsync(void** devPtr, size_t size, musaStream_t hStream);

__host__ musaError_t musaFreeAsync(void* devPtr, musaStream_t hStream);

__host__ musaError_t
musaDeviceGetDefaultMemPool(musaMemPool_t* memPool, int device);

__host__ musaError_t musaMemPoolGetAttribute(
    musaMemPool_t memPool,
    enum musaMemPoolAttr attr,
    void* value);

__host__ musaError_t musaMemPoolSetAttribute(
    musaMemPool_t memPool,
    enum musaMemPoolAttr attr,
    void* value);

} // namespace musa

} // namespace c10

#endif // TORCH_MUSA_CSRC_ATEN_OPS_MUSA_UNIMPLEMENTED_FUNCTIONS_H
