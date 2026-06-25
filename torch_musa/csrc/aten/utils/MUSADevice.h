#ifndef TORCH_MUSA_CSRC_ATEN_UTILS_MUSA_DEVICE_H_
#define TORCH_MUSA_CSRC_ATEN_UTILS_MUSA_DEVICE_H_

#include <c10/core/Backend.h>
#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>

namespace at::musa {

constexpr c10::Backend kMUSABackend = c10::Backend::PrivateUse1;
constexpr c10::DeviceType kMUSA = c10::DeviceType::PrivateUse1;
constexpr c10::DispatchKey kMUSAKey = c10::DispatchKey::PrivateUse1;

} // namespace at::musa

namespace at {
using musa::kMUSA;
} // namespace at

#endif // TORCH_MUSA_CSRC_ATEN_UTILS_MUSA_DEVICE_H_
