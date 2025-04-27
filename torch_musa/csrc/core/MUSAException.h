#ifndef TORCH_MUSA_CSRC_CORE_MUSAEXCEPTION_H_
#define TORCH_MUSA_CSRC_CORE_MUSAEXCEPTION_H_
#include <c10/util/Exception.h>
#include <musa.h>
#include <musa_runtime_api.h>

#define TORCH_MUSA_CHECK(EXPR)                                       \
  do {                                                               \
    const musaError_t __err = EXPR;                                  \
    if (__err != musaSuccess) {                                      \
      TORCH_CHECK(false, "MUSA error: ", musaGetErrorString(__err)); \
    }                                                                \
  } while (0)

#define TORCH_MUSA_CHECK_WARN(EXPR)                            \
  do {                                                         \
    const musaError_t __err = EXPR;                            \
    if (C10_UNLIKELY(__err != musaSuccess)) {                  \
      auto error_unused = musaGetLastError();                  \
      (void)error_unused;                                      \
      TORCH_WARN("MUSA warning: ", musaGetErrorString(__err)); \
    }                                                          \
  } while (0)

#define TORCH_MUSA_WARN(EXPR) TORCH_MUSA_CHECK_WARN(EXPR)
#define C10_MUSA_CHECK(EXPR)                                        \
  do {                                                              \
    const musaError_t __err = EXPR;                                 \
    c10::musa::c10_musa_check_implementation(                       \
        static_cast<int32_t>(__err),                                \
        __FILE__,                                                   \
        __func__, /* Line number data type not well-defined between \
                      compilers, so we perform an explicit cast */  \
        static_cast<uint32_t>(__LINE__),                            \
        true);                                                      \
  } while (0)

#define C10_MUSA_CHECK_WARN(EXPR)                              \
  do {                                                         \
    const musaError_t __err = EXPR;                            \
    if (C10_UNLIKELY(__err != musaSuccess)) {                  \
      auto error_unused C10_UNUSED = musaGetLastError();       \
      (void)error_unused;                                      \
      TORCH_WARN("MUSA warning: ", musaGetErrorString(__err)); \
    }                                                          \
  } while (0)

#define TORCH_MUSA_ERROR_HANDLE(EXPR) EXPR

#define C10_MUSA_KERNEL_LAUNCH_CHECK() TORCH_MUSA_CHECK(musaGetLastError())

// Indicates that a MUSA error is handled in a non-standard way
#define C10_MUSA_ERROR_HANDLED(EXPR) EXPR

// Intentionally ignore a MUSA error
#define C10_MUSA_IGNORE_ERROR(EXPR)                             \
  do {                                                          \
    const musaError_t __err = EXPR;                             \
    if (C10_UNLIKELY(__err != musaSuccess)) {                   \
      musaError_t error_unused C10_UNUSED = musaGetLastError(); \
      (void)error_unused;                                       \
    }                                                           \
  } while (0)

#define CHECK_MUDNN_STATUS(rst, msg)       \
  TORCH_CHECK(                             \
      rst == ::musa::dnn::Status::SUCCESS, \
      __FUNCTION__,                        \
      " MUDNN failed in: ",                \
      msg);

namespace c10 {
namespace musa {
void c10_musa_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const int line_number,
    const bool include_device_assertions);
} // namespace musa
} // namespace c10

#endif // TORCH_MUSA_CSRC_CORE_MUSAEXCEPTION_H_
