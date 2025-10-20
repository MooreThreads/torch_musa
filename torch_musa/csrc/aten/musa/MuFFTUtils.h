#pragma once

#include <ATen/Config.h>

#include <mufft.h>
#include <mufftXt.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace at {
namespace native {

// This means that max dim is 3 + 2 = 5 with batch dimension and possible
// complex dimension
constexpr int max_rank = 3;

static inline std::string _musaGetErrorEnum(mufftResult error) {
  switch (error) {
    case MUFFT_SUCCESS:
      return "MUFFT_SUCCESS";
    case MUFFT_INVALID_PLAN:
      return "MUFFT_INVALID_PLAN";
    case MUFFT_ALLOC_FAILED:
      return "MUFFT_ALLOC_FAILED";
    case MUFFT_INVALID_TYPE:
      return "MUFFT_INVALID_TYPE";
    case MUFFT_INVALID_VALUE:
      return "MUFFT_INVALID_VALUE";
    case MUFFT_INTERNAL_ERROR:
      return "MUFFT_INTERNAL_ERROR";
    case MUFFT_EXEC_FAILED:
      return "MUFFT_EXEC_FAILED";
    case MUFFT_SETUP_FAILED:
      return "MUFFT_SETUP_FAILED";
    case MUFFT_INVALID_SIZE:
      return "MUFFT_INVALID_SIZE";
    case MUFFT_UNALIGNED_DATA:
      return "MUFFT_UNALIGNED_DATA";
    case MUFFT_INCOMPLETE_PARAMETER_LIST:
      return "MUFFT_INCOMPLETE_PARAMETER_LIST";
    case MUFFT_INVALID_DEVICE:
      return "MUFFT_INVALID_DEVICE";
    case MUFFT_PARSE_ERROR:
      return "MUFFT_PARSE_ERROR";
    case MUFFT_NO_WORKSPACE:
      return "MUFFT_NO_WORKSPACE";
    case MUFFT_NOT_IMPLEMENTED:
      return "MUFFT_NOT_IMPLEMENTED";
#if !defined(USE_ROCM)
    case MUFFT_LICENSE_ERROR:
      return "MUFFT_LICENSE_ERROR";
#endif
    case MUFFT_NOT_SUPPORTED:
      return "MUFFT_NOT_SUPPORTED";
    default:
      std::ostringstream ss;
      ss << "unknown error " << error;
      return ss.str();
  }
}

static inline void MUFFT_CHECK(mufftResult error) {
  if (error != MUFFT_SUCCESS) {
    std::ostringstream ss;
    ss << "muFFT error: " << _musaGetErrorEnum(error);
    AT_ERROR(ss.str());
  }
}

} // namespace native
} // namespace at
