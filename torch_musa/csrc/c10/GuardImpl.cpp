#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#pragma GCC diagnostic pop

namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(
    MTGPU,
    c10::impl::NoOpDeviceGuardImpl<DeviceType::MTGPU>);

} // namespace detail
} // namespace at
