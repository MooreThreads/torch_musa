""" MUSA Kernel Device codes """

import torch


def musa_kernel_driver() -> str:
    """
    musa kernel check/load/launch c++ code.
    """

    cap = torch.musa.get_device_capability()

    source_codes = """
            #define MUSA_DRIVER_CHECK(EXPR)                    \\
            do {                                               \\
                MUresult code = EXPR;                          \\
                const char *msg;                               \\
                muGetErrorString(code, &msg);                  \\
                if (code != MUSA_SUCCESS) {                    \\
                    throw std::runtime_error(                  \\
                        std::string("MUSA driver error: ") +   \\
                        std::string(msg));                     \\
                }                                              \\
            } while (0);

            namespace {

            struct Grid {
                Grid(uint32_t x, uint32_t y, uint32_t z)
                  : grid_x(x), grid_y(y), grid_z(z) {}
                uint32_t grid_x;
                uint32_t grid_y;
                uint32_t grid_z;

                bool is_non_zero() {
                    return grid_x > 0 && grid_y > 0 && grid_z > 0;
                }
            };

            }  // anonymous namespace

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
                    func, gridX, gridY, gridZ, 128*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
                ));
            }
            """
    if cap[0] > 2:
        source_codes = source_codes.replace("128*numWarps", "32*numWarps")
    return source_codes


def musa_kernel_header() -> str:
    """
    musa kernel c++ header code.
    """
    source_codes = """
                #include "torch_musa/csrc/aten/ops/musa/unimplemented_functions.h"
                #include "torch_musa/csrc/core/MUSAGuard.h"
                #include "torch_musa/csrc/core/MUSAStream.h"
                #include "torch_musa/csrc/aten/ops/TensorFactory.h"
                """
    return source_codes
