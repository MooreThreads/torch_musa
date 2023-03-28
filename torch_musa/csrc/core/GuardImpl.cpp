#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(
    PrivateUse1,
    c10::impl::NoOpDeviceGuardImpl<::at::native::musa::kMUSA>);

} // namespace detail
} // namespace at
