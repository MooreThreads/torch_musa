# pylint: disable=C0103,C0116

"""MUSADeviceOpOverrides implementation"""
# pylint: disable=C0103, C0116

from typing import Optional
from torch._inductor.utils import triton_version_uses_attrs_dict

# pylint:disable=unused-import
import torch
from torch._inductor.codegen.common import (
    DeviceOpOverrides,
    register_device_op_overrides,
)


class MUSADeviceOpOverrides(DeviceOpOverrides):
    """class of MUSADeviceOpOverrides"""

    def import_get_raw_stream_as(self, name):
        return f"from torch_musa._MUSAC import _musa_getCurrentRawStream as {name}"

    def set_device(self, device_idx):
        return f"torch.musa.set_device({device_idx})"

    def synchronize(self):
        return "torch.musa.synchronize()"

    def device_guard(self, device_idx):
        return f"torch.musa._DeviceGuard({device_idx})"

    def cpp_device_guard(self) -> str:
        return "c10::musa::MUSAGuard"

    def cpp_aoti_device_guard(self) -> str:
        return "AOTIMusaGuard"

    def cpp_stream_guard(self) -> str:
        return "c10::musa::MUSAStreamGuard"

    def cpp_aoti_stream_guard(self) -> str:
        return "AOTIMusaStreamGuard"

    def cpp_getStreamFromExternal(self) -> str:
        return "c10::musa::getStreamFromExternal"

    def kernel_header(self) -> str:
        source_codes = """
        #include <c10/musa/MUSAGuard.h>
        #include <c10/musa/MUSAStream.h>
        #include <ATen/musa/EmptyTensor.h>
        """
        return source_codes

    def kernel_driver(self) -> str:
        cap = torch.musa.get_device_capability()
        source_codes = """
            #define MUSA_DRIVER_CHECK(EXPR)                    \\
            do {                                               \\
                MUresult code = EXPR;                          \\
                const char *msg;                               \\
                MUresult code_get_error = muGetErrorString(code, &msg); \\
                if (code_get_error != MUSA_SUCCESS) {          \\
                    throw std::runtime_error(                  \\
                        std::string("MUSA driver error: ") +   \\
                        std::string("invalid error code!"));   \\
                }                                              \\
                if (code != MUSA_SUCCESS) {                    \\
                    throw std::runtime_error(                  \\
                        std::string("MUSA driver error: ") +   \\
                        std::string(msg));                     \\
                }                                              \\
            } while (0);

            static inline MUfunction loadKernel(
                    std::string filePath,
                    const std::string &funcName,
                    uint32_t sharedMemBytes,
                    const std::optional<std::string> &mubinDir = std::nullopt) {
                if (mubinDir) {
                    std::filesystem::path p1{*mubinDir};
                    std::filesystem::path p2{filePath};
                    filePath = (p1 / p2.filename()).string();
                }

                MUmodule mod;
                MUfunction func;
                MUSA_DRIVER_CHECK(muModuleLoad(&mod, filePath.c_str()));
                MUSA_DRIVER_CHECK(muModuleGetFunction(&func, mod, funcName.c_str()));
                if (sharedMemBytes > 0) {
                    MUSA_DRIVER_CHECK(muFuncSetAttribute(
                        func,
                        MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        sharedMemBytes
                    ))
                }
                return func;
            }

            static inline void launchKernel(
                    MUfunction func,
                    uint32_t gridX,
                    uint32_t gridY,
                    uint32_t gridZ,
                    uint32_t numWarps,
                    uint32_t sharedMemBytes,
                    void* args[],
                    musaStream_t stream) {
                MUSA_DRIVER_CHECK(muLaunchKernel(
                    func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
                ));
            }
        """
        if cap[0] > 2:
            source_codes = source_codes.replace("128*numWarps", "32*numWarps")
        return source_codes

    def tma_descriptor_helpers(self) -> str:

        # helper functions for initializing 1D and 2D TMA descriptors in C++.
        # borrowed from the Triton code here:
        # https://github.com/triton-lang/triton/blob/6af4f88591c85de079d8a36a4d7dba67918e2b39/third_party/nvidia/backend/driver.c#L283
        return """
            #if !defined(USE_ROCM) && defined(MUSA_VERSION) && MUSA_VERSION >= 310
            [[maybe_unused]] static void init1DTMADescriptor(
                    MUtensorMap* m,
                    void* globalAddress,
                    uint64_t dim,
                    uint32_t blockDim,
                    uint32_t elementSize) {
                uint64_t dims[1] = {dim};
                uint64_t globalStrides[1] = {dim * elementSize};
                uint32_t tensorDims[1] = {blockDim};
                uint32_t elementStrides[1] = {1};

                MUtensorMapDataType type;
                switch (elementSize) {
                case 1:
                    type = MU_TENSOR_MAP_DATA_TYPE_UINT8;
                    break;
                case 2:
                    type = MU_TENSOR_MAP_DATA_TYPE_UINT16;
                    break;
                case 4:
                    type = MU_TENSOR_MAP_DATA_TYPE_UINT32;
                    break;
                default:
                    throw std::runtime_error("elementSize must be 1, 2, or 4");
                }

                if (elementSize * blockDim < 32) {
                    throw std::runtime_error("block size too small");
                }

                int rank = 1;

                MUSA_DRIVER_CHECK(muTensorMapEncodeTiled(
                    m, type, rank, globalAddress, dims,
                    globalStrides, tensorDims, elementStrides, MU_TENSOR_MAP_INTERLEAVE_NONE,
                    MU_TENSOR_MAP_SWIZZLE_NONE, MU_TENSOR_MAP_L2_PROMOTION_NONE,
                    MU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
            }

            [[maybe_unused]] static void init2DTMADescriptor(
                    MUtensorMap* m,
                    void* globalAddress,
                    uint64_t dim1,
                    uint64_t dim0,
                    uint32_t blockDim1,
                    uint32_t blockDim0,
                    uint32_t elementSize) {
                uint64_t dims[2] = {dim0, dim1};
                uint32_t tensorDims[2] = {blockDim0, blockDim1};
                uint64_t globalStrides[2] = {dims[0] * elementSize,
                                             dims[0] * dims[1] * elementSize};
                uint32_t elementStrides[2] = {1, 1};

                MUtensorMapDataType type;
                switch (elementSize) {
                case 1:
                    type = MU_TENSOR_MAP_DATA_TYPE_UINT8;
                    break;
                case 2:
                    type = MU_TENSOR_MAP_DATA_TYPE_UINT16;
                    break;
                case 4:
                    type = MU_TENSOR_MAP_DATA_TYPE_UINT32;
                    break;
                default:
                    throw std::runtime_error("elementSize must be 1, 2, or 4");
                }

                int rank = 2;

                MUtensorMapSwizzle swizzle = MU_TENSOR_MAP_SWIZZLE_128B;
                uint32_t contigDimSizeInByte = elementSize * tensorDims[0];
                if (contigDimSizeInByte >= 128) {
                    swizzle = MU_TENSOR_MAP_SWIZZLE_128B;
                } else if (contigDimSizeInByte >= 64) {
                    swizzle = MU_TENSOR_MAP_SWIZZLE_64B;
                } else if (contigDimSizeInByte >= 32) {
                    swizzle = MU_TENSOR_MAP_SWIZZLE_32B;
                } else {
                    throw std::runtime_error("block size too small");
                }

                if (contigDimSizeInByte > 128) {
                    tensorDims[0] = 128 / elementSize;
                }

                MUSA_DRIVER_CHECK(muTensorMapEncodeTiled(
                    m, type, rank, globalAddress, dims,
                    globalStrides, tensorDims, elementStrides, MU_TENSOR_MAP_INTERLEAVE_NONE,
                    swizzle, MU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                    MU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
            }
            #endif
        """

    def cpp_stream_type(self) -> str:
        return "musaStream_t"

    def aoti_get_stream(self) -> str:
        return "aoti_torch_get_current_musa_stream"

    def cpp_kernel_type(self) -> str:
        return "MUfunction"

    def cpp_device_ptr(self) -> str:
        return "MUdeviceptr"

    def cpp_global_scratch(self, idx: int) -> Optional[tuple[str, str]]:
        if triton_version_uses_attrs_dict():
            return f"MUdeviceptr global_scratch_{idx} = 0;", f"global_scratch_{idx}"
        return None


register_device_op_overrides("musa", MUSADeviceOpOverrides())
