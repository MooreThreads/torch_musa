#pragma once
// Light-weight version of CUDAContext.h with fewer transitive includes

#include <cstdint>

#include <mublas.h>
#include <musa_runtime_api.h>
#include <musparse.h>

// TODO(@sw-compute): mublasLT has not been developed
// #include <mublasLt.h>

// TODO(@sw-compute): mudss has not been developed
// #if defined(USE_CUDSS)
// #include <cudss.h>
// #endif

#include <c10/core/Allocator.h>
// #include "torch_musa/csrc/core/MUSAFunctions.h"

#include <mublas.h>

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION >= 4030)
#define USE_MUBLASLT 1
#include <mublasLt.h>
#else
#define USE_MUBLASLT 0
typedef struct {
} * mublasLtHandle_t;
#endif

namespace c10 {
struct Allocator;
}

namespace at::musa {

TORCH_CUDA_CPP_API musaDeviceProp* getCurrentDeviceProperties();

TORCH_CUDA_CPP_API int warp_size();

TORCH_CUDA_CPP_API musaDeviceProp* getDeviceProperties(c10::DeviceIndex device);

TORCH_CUDA_CPP_API bool canDeviceAccessPeer(
    c10::DeviceIndex device,
    c10::DeviceIndex peer_device);

TORCH_CUDA_CPP_API c10::Allocator* getMUSADeviceAllocator();

/* Handles */
TORCH_CUDA_CPP_API musparseHandle_t getCurrentCUDASparseHandle();
TORCH_CUDA_CPP_API mublasHandle_t getCurrentMUSABlasHandle();
#if USE_MUBLASLT
TORCH_CUDA_CPP_API mublasLtHandle_t getCurrentMUSABlasLtHandle();
#endif
TORCH_CUDA_CPP_API void clearCublasWorkspaces();

// TODO(@sw-compute): musolverDn has not been developed
#if defined(MUSART_VERSION)
// TORCH_CUDA_CPP_API musolverDnHandle_t getCurrentCUDASolverDnHandle();
#endif

// TODO(@sw-compute): mudss has not been developed
// #if defined(USE_CUDSS)
// TORCH_CUDA_CPP_API cudssHandle_t getCurrentCudssHandle();
// #endif

} // namespace at::musa
