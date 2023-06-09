#include <ATen/AccumulateType.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/adaption.h>

#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <mudnn.h>

namespace at {
namespace musa {

Tensor& ArangeStartOut(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result) {
  MUSA_TENSOR_TYPE_CHECK(result);
  c10::musa::MUSAGuard device_guard(result.device());
  double size_d =
      std::ceil((end.toDouble() - start.toDouble()) / step.toDouble());
  int64_t size = static_cast<int64_t>(size_d);
  std::vector<int64_t> shape{size};
  result.resize_(shape);
  auto out = CreateMUTensor(result);
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Arange op;
  if (result.scalar_type() == at::ScalarType::Float) {
    CHECK_MUDNN_STATUS(op.SetStart(start.toDouble()), "SetStart");
    CHECK_MUDNN_STATUS(op.SetStep(step.toDouble()), "SetStep");
  } else {
    CHECK_MUDNN_STATUS(op.SetStart(start.toLong()), "SetStart");
    CHECK_MUDNN_STATUS(op.SetStep(step.toLong()), "SetStep");
  }
  CHECK_MUDNN_STATUS(op.SetEnd(end.toDouble()), "SetEnd");

  CHECK_MUDNN_STATUS(op.Run(h, out), "Run ");
  return result;
}

Tensor& ArangeOut(const Scalar& end, Tensor& result) {
  return ArangeStartOut(/*start=*/0, end, /*step*/ 1, result);
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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("arange.out", &ArangeOut);
  m.impl("arange.start_out", &ArangeStartOut);
  m.impl("linspace.out", &LinspaceOut);
}

} // namespace musa
} // namespace at
