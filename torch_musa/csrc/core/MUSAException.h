#ifndef TORCH_MUSA_CSRC_CORE_MUSA_MUSAEXCEPTION_H_
#define TORCH_MUSA_CSRC_CORE_MUSA_MUSAEXCEPTION_H_
#include <c10/util/Exception.h>
#include "musa_runtime_api.h"

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
