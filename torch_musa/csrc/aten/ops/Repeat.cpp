#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {
Tensor RepeatInterleaveTensor(
    const Tensor& repeats,
    c10::optional<int64_t> output_size) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, repeats, "RepeatInterleaveTensor", "repeats");
  const OptionalDeviceGuard device_guard(device_of(repeats));
  return at::native::repeat_interleave_cuda(repeats, output_size);
}

} // namespace musa
} // namespace at
