#ifndef TORCH_MUSA_CSRC_CORE_MUSA_MUSAEXCEPTION_H_
#define TORCH_MUSA_CSRC_CORE_MUSA_MUSAEXCEPTION_H_
#include "musa_runtime_api.h"

#define TORCH_MUSARUNTIME_CHECK(EXPR)                                \
  do {                                                               \
    musaError_t __err = EXPR;                                        \
    if (__err != musaSuccess) {                                      \
      TORCH_CHECK(false, "MUSA error: ", musaGetErrorString(__err)); \
    }                                                                \
  } while (0)

#define CHECK_MUDNN_STATUS(rst, msg)       \
  TORCH_CHECK(                             \
      rst == ::musa::dnn::Status::SUCCESS, \
      __FUNCTION__,                        \
      " MUDNN failed in: ",                \
      msg);

#endif // TORCH_MUSA_CSRC_CORE_MUSA_MUSAEXCEPTION_H_
