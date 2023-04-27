#ifndef TORCH_MUSA_CSRC_CORE_MUSA_MUSAEXCEPTION_H_
#define TORCH_MUSA_CSRC_CORE_MUSA_MUSAEXCEPTION_H_
#include <c10/util/Exception.h>
#include "musa_runtime_api.h"

namespace torch_musa {
class MuDNNError : public c10::Error {
  using c10::Error::Error;
};
} // namespace torch_musa

#define TORCH_MUDNN_CHECK_WITH_SHAPES(EXPR, ...) \
  AT_MUDNN_CHECK(EXPR, "\n", ##__VA_ARGS__)

// See Note [CHECK macro]
#define TORHC_MUDNN_CHECK(EXPR, ...)                                            \
  do {                                                                          \
    mudnnStatus_t status = EXPR;                                                \
    if (status != MUDNN_STATUS_SUCCESS) {                                       \
      if (status == MUDNN_STATUS_NOT_SUPPORTED) {                               \
        TORCH_CHECK_WITH(                                                       \
            MuDNNError,                                                         \
            false,                                                              \
            "muDNN error: ",                                                    \
            mudnnGetErrorString(status),                                        \
            ". This error may appear if you passed in a non-contiguous input.", \
            ##__VA_ARGS__);                                                     \
      } else {                                                                  \
        TORCH_CHECK_WITH(                                                       \
            MuDNNError,                                                         \
            false,                                                              \
            "muDNN error: ",                                                    \
            mudnnGetErrorString(status),                                        \
            ##__VA_ARGS__);                                                     \
      }                                                                         \
    }                                                                           \
  } while (0)

#define TORCH_MUSA_CHECK(EXPR)                                       \
  do {                                                               \
    const musaError_t __err = EXPR;                                  \
    if (__err != musaSuccess) {                                      \
      TORCH_CHECK(false, "MUSA error: ", musaGetErrorString(__err)); \
    }                                                                \
  } while (0)

#define TORCH_MUSA_WARN(EXPR)                                  \
  do {                                                         \
    const musaError_t __err = EXPR;                            \
    if (C10_UNLIKELY(__err != musaSuccess)) {                  \
      TORCH_WARN("MUSA warning: ", musaGetErrorString(__err)); \
    }                                                          \
  } while (0)

#define TORCH_MUSA_ERROR_HANDLE(EXPR) EXPR

#define CHECK_MUDNN_STATUS(rst, msg)       \
  TORCH_CHECK(                             \
      rst == ::musa::dnn::Status::SUCCESS, \
      __FUNCTION__,                        \
      " MUDNN failed in: ",                \
      msg);

#endif // TORCH_MUSA_CSRC_CORE_MUSA_MUSAEXCEPTION_H_
