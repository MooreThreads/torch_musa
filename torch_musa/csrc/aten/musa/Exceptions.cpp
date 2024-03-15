#include "torch_musa/csrc/aten/musa/Exceptions.h"

#include <mublas.h>

namespace at {
namespace musa {
namespace blas {

const char* _mublasGetErrorEnum(mublasStatus_t error) {
  switch (error) {
    case MUBLAS_STATUS_SUCCESS:
      return "MUBLAS_STATUS_SUCCESS";
    case MUBLAS_STATUS_INVALID_HANDLE:
      return "MUBLAS_STATUS_INVALID_HANDLE";
    case MUBLAS_STATUS_NOT_IMPLEMENTED:
      return "MUBLAS_STATUS_NOT_IMPLEMENTED";
    case MUBLAS_STATUS_INVALID_POINTER:
      return "MUBLAS_STATUS_INVALID_POINTER";
    case MUBLAS_STATUS_INVALID_SIZE:
      return "MUBLAS_STATUS_INVALID_SIZE";
    case MUBLAS_STATUS_MEMORY_ERROR:
      return "MUBLAS_STATUS_MEMORY_ERROR";
    case MUBLAS_STATUS_INTERNAL_ERROR:
      return "MUBLAS_STATUS_INTERNAL_ERROR";
    case MUBLAS_STATUS_PERF_DEGRADED:
      return "MUBLAS_STATUS_PERF_DEGRADED";
    case MUBLAS_STATUS_SIZE_QUERY_MISMATCH:
      return "MUBLAS_STATUS_SIZE_QUERY_MISMATCH";
    case MUBLAS_STATUS_SIZE_INCREASED:
      return "MUBLAS_STATUS_SIZE_INCREASED";
    case MUBLAS_STATUS_SIZE_UNCHANGED:
      return "MUBLAS_STATUS_SIZE_UNCHANGED";
    case MUBLAS_STATUS_INVALID_VALUE:
      return "MUBLAS_STATUS_INVALID_VALUE";
    case MUBLAS_STATUS_CONTINUE:
      return "MUBLAS_STATUS_CONTINUE";
    case MUBLAS_STATUS_CHECK_NUMERICS_FAIL:
      return "MUBLAS_STATUS_CHECK_NUMERICS_FAIL";
    default:
      return "<unknown>";
  }
}

} // namespace blas
} // namespace musa
} // namespace at
