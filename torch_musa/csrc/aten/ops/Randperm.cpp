#include <ATen/Config.h>
// clang-format off
// Some classes in NativeFunctions.h require the corrosponding definition in Exception.h
#include <c10/util/Exception.h>
// clang-format on
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Activation.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorConversions.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace musa {

namespace {

at::Tensor& RandpermGeneratorOut(
    int64_t n,
    c10::optional<at::Generator> generator,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_CUDA_generator_out_randperm_out", "out");
  const OptionalDeviceGuard device_guard(device_of(out));
  return at::native::randperm_out_cuda(n, generator, out);
}

} // anonymous namespace

ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "randperm.generator_out",
    RandpermGeneratorOut)

} // namespace musa
} // namespace at
