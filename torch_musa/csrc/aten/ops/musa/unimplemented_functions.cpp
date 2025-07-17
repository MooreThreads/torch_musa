#include <c10/util/Exception.h>

#include <musa_runtime_api.h>

#include "torch_musa/csrc/aten/ops/musa/unimplemented_functions.h"

namespace c10 {
namespace musa {

__host__ musaError_t
musaThreadExchangeStreamCaptureMode(enum musaStreamCaptureMode* mode) {
  C10_THROW_ERROR(
      NotImplementedError,
      "musaThreadExchangeStreamCaptureMode is not supported now!");
  return musaErrorNotSupported;
}

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

} // namespace musa
} // namespace c10
