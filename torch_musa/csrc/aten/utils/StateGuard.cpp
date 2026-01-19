#include "torch_musa/csrc/aten/utils/StateGuard.h"

namespace at::musa {

#define DECL_STATE(NAME, TYPE, INIT) inline thread_local TYPE NAME = INIT;

DECL_STATE(STRICT_MASK_SELECT, bool, false)

#define DEFINE_GUARD(NAME, TYPE)                             \
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

DEFINE_GUARD(STRICT_MASK_SELECT, bool);

#undef DEFINE_GUARD
#undef DECL_STATE

} // namespace at::musa
