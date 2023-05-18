#ifndef TORCH_MUSA_CSRC_ATEN_MUSA_EXCEPTIONS_H_
#define TORCH_MUSA_CSRC_ATEN_MUSA_EXCEPTIONS_H_

#include <mublas.h>

#include "torch_musa/csrc/core/MUSAException.h"

namespace at {
namespace musa {
namespace blas {

const char* _mublasGetErrorEnum(mublasStatus_t error);

#define TORCH_MUSABLAS_CHECK(EXPR)                  \
  do {                                              \
    mublasStatus_t __err = EXPR;                    \
    TORCH_CHECK(                                    \
        __err == MUBLAS_STATUS_SUCCESS,             \
        "MUSA error: ",                             \
        at::musa::blas::_mublasGetErrorEnum(__err), \
        " when calling `" #EXPR "`");               \
  } while (0)

#define AT_MUSA_CHECK(EXPR) TORCH_MUSA_CHECK(EXPR)

} // namespace blas
} // namespace musa
} // namespace at
#endif // TORCH_MUSA_CSRC_ATEN_MUSA_EXCEPTIONS_H_
