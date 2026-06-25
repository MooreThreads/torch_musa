#include "torch_musa/csrc/aten/utils/StateGuard.h"

namespace at::musa {

#define DECL_STATE(NAME, TYPE, INIT) inline thread_local TYPE NAME = INIT;

#define DEFINE_GUARD(NAME, TYPE, INIT)                       \
  DECL_STATE(NAME, TYPE, INIT)                               \
  STATE_GUARD_CLS(NAME)::STATE_GUARD_CLS(NAME)(TYPE state) { \
    prev_ = NAME;                                            \
    NAME = state;                                            \
  }                                                          \
  STATE_GUARD_CLS(NAME)::~STATE_GUARD_CLS(NAME)() {          \
    NAME = prev_;                                            \
  }                                                          \
  TYPE Get##NAME##State() {                                  \
    return NAME;                                             \
  }

DEFINE_GUARD(STRICT_MASK_SELECT, bool, false);
DEFINE_GUARD(NNZ_NONE_CONTIGUOUS, bool, false);

#undef DEFINE_GUARD
#undef DECL_STATE

} // namespace at::musa
