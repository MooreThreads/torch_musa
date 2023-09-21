#include <ATen/AccumulateType.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/adaption.h>

#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

#include <mudnn.h>
namespace at {
namespace musa {

// TODO(zaixing.wang): fp16 mark
Tensor& ArangeStartOut(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  c10::musa::MUSAGuard device_guard(out.device());
  return at::native::arange_cuda_out(start, end, step, out);
}

Tensor& ArangeOut(const Scalar& end, Tensor& result) {
  return ArangeStartOut(/*start=*/0, end, /*step*/ 1, result);
}

Tensor& range_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result) {
  c10::musa::MUSAGuard device_guard(result.device());
  return at::native::range_cuda_out(start, end, step, result);
}

at::Tensor& LinspaceOut(
    const at::Scalar& start,
    const at::Scalar& end,
    int64_t steps,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, out, "LinspaceOut", "out");
  const OptionalDeviceGuard device_guard(device_of(out));
  return at::native::linspace_cuda_out(start, end, steps, out);
}

ADVANCED_REGISTER(aten, PrivateUse1, "arange.start_out", ArangeStartOut)
ADVANCED_REGISTER(aten, PrivateUse1, "linspace.out", LinspaceOut)
ADVANCED_REGISTER(aten, PrivateUse1, "range.out", range_out)

} // namespace musa
} // namespace at
