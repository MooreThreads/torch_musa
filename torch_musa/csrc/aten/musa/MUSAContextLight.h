#pragma once
// Light-weight version of CUDAContext.h with fewer transitive includes

#include <cstdint>

#include <mublas.h>
#include <musa_runtime_api.h>
#include <musparse.h>

// TODO(@sw-compute): mublasLT has not been developed
// #include <mublasLt.h>

#if 0
#include <musolverDn.h>
#endif

// TODO(@sw-compute): mudss has not been developed
// #if defined(USE_CUDSS)
// #include <cudss.h>
// #endif

#if defined(USE_ROCM)
#include <hipsolver/hipsolver.h>
#endif

#include <c10/core/Allocator.h>
#include "torch_musa/csrc/core/MUSAFunctions.h"

namespace c10 {
struct Allocator;
}

namespace at::musa {

/*
A common CUDA interface for ATen.

This interface is distinct from CUDAHooks, which defines an interface that links
to both CPU-only and CUDA builds. That interface is intended for runtime
dispatch and should be used from files that are included in both CPU-only and
CUDA builds.

CUDAContext, on the other hand, should be preferred by files only included in
CUDA builds. It is intended to expose CUDA functionality in a consistent
manner.

This means there is some overlap between the CUDAContext and CUDAHooks, but
the choice of which to use is simple: use CUDAContext when in a CUDA-only file,
use CUDAHooks otherwise.

Note that CUDAContext simply defines an interface with no associated class.
It is expected that the modules whose functions compose this interface will
manage their own state. There is only a single CUDA context/state.
*/

/**
 * DEPRECATED: use device_count() instead
 */
inline int64_t getNumGPUs() {
  return c10::musa::device_count();
}

/**
 * CUDA is available if we compiled with CUDA, and there are one or more
 * devices.  If we compiled with CUDA but there is a driver problem, etc.,
 * this function will report CUDA is not available (rather than raise an error.)
 */
inline bool is_available() {
  return c10::musa::device_count() > 0;
}

TORCH_CUDA_CPP_API musaDeviceProp* getCurrentDeviceProperties();

TORCH_CUDA_CPP_API int warp_size();

TORCH_CUDA_CPP_API musaDeviceProp* getDeviceProperties(c10::DeviceIndex device);

TORCH_CUDA_CPP_API bool canDeviceAccessPeer(
    c10::DeviceIndex device,
    c10::DeviceIndex peer_device);

TORCH_CUDA_CPP_API c10::Allocator* getMUSADeviceAllocator();

/* Handles */
TORCH_CUDA_CPP_API musparseHandle_t getCurrentCUDASparseHandle();
TORCH_CUDA_CPP_API mublasHandle_t getCurrentCUDABlasHandle();
// TORCH_CUDA_CPP_API mublasLtHandle_t getCurrentCUDABlasLtHandle();

TORCH_CUDA_CPP_API void clearCublasWorkspaces();

#if defined(MUSART_VERSION) || defined(USE_ROCM)
// TORCH_CUDA_CPP_API musolverDnHandle_t getCurrentCUDASolverDnHandle();
#endif

// TODO(@sw-compute): mudss has not been developed
// #if defined(USE_CUDSS)
// TORCH_CUDA_CPP_API cudssHandle_t getCurrentCudssHandle();
// #endif

} // namespace at::musa
