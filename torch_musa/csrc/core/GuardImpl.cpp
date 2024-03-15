
#include "torch_musa/csrc/core/GuardImpl.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace c10 {
namespace musa {
namespace impl {

constexpr DeviceType MUSAGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(PrivateUse1, MUSAGuardImpl);

} // namespace impl
} // namespace musa
} // namespace c10
