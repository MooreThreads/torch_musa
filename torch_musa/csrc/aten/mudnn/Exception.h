#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_EXCEPTION_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_EXCEPTION_H_

#include <c10/util/Exception.h>

#include <mudnn.h>

namespace c10 {

using MuDNNError = c10::Error;

} // namespace c10

// clang-format off

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#define AT_MUDNN_CHECK_WITH_SHAPES(EXPR, ...) AT_MUDNN_CHECK(EXPR, "\n", ##__VA_ARGS__)

#define AT_MUDNN_CHECK(EXPR, ...)                                                               \
  do {                                                                                          \
    mudnnStatus_t status = EXPR;                                                                \
    if (status != MUDNN_STATUS_SUCCESS) {                                                       \
      if (status == MUDNN_STATUS_NOT_SUPPORTED) {                                               \
        TORCH_CHECK_WITH(MuDNNError, false,                                                     \
            "muDNN error: ",                                                                    \
            mudnnGetErrorString(status),                                                        \
            ". This error may appear if you passed in a non-contiguous input.", ##__VA_ARGS__); \
      } else {                                                                                  \
        TORCH_CHECK_WITH(MuDNNError, false,                                                     \
            "muDNN error: ", mudnnGetErrorString(status), ##__VA_ARGS__);                       \
      }                                                                                         \
    }                                                                                           \
  } while (0)

#define CHECK_MUDNN_STATUS(rst, msg) \
  TORCH_CHECK(                       \
      rst == MUDNN_STATUS_SUCCESS,   \
      __FUNCTION__,                  \
      " MUDNN failed in: ",          \
      msg);

#else

#define CHECK_MUDNN_STATUS(rst, msg)       \
  TORCH_CHECK(                             \
      rst == ::musa::dnn::Status::SUCCESS, \
      __FUNCTION__,                        \
      " MUDNN failed in: ",                \
      msg);

#endif

// clang-format on

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_EXCEPTION_H_
