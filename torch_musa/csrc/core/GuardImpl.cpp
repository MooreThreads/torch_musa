#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "torch_musa/csrc/core/GuardImpl.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace torch_musa {
namespace impl {

constexpr DeviceType MUSAGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(PrivateUse1, MUSAGuardImpl);

} // namespace impl
} // namespace torch_musa
