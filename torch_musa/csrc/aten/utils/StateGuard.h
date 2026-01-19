#ifndef TORCH_MUSA_CSRC_ATEN_UTILS_STATEGUARD_H_
#define TORCH_MUSA_CSRC_ATEN_UTILS_STATEGUARD_H_

#include <c10/macros/Macros.h>

namespace at::musa {

#define STATE_GUARD_CLS(NAME) NAME_##StateGuard

#define DECL_STATE_GUARD(NAME, TYPE, INIT) \
  struct STATE_GUARD_CLS(NAME) {           \
    TYPE prev_ = INIT;                     \
    STATE_GUARD_CLS(NAME)(TYPE state);     \
    ~STATE_GUARD_CLS(NAME)();              \
  };                                       \
  TYPE Get##NAME##State();

DECL_STATE_GUARD(STRICT_MASK_SELECT, bool, false);

#undef DECL_STATE_GUARD

#define _MAKE_STATE_GUARD(NAME, UID, STATE) \
  const STATE_GUARD_CLS(NAME) C10_CONCATENATE(NAME, UID)(STATE);

#define MAKE_STATE_GUARD(NAME, STATE) _MAKE_STATE_GUARD(NAME, C10_UID, STATE)

#define GET_STATE(NAME) Get##NAME##State()

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_UTILS_STATEGUARD_H_
