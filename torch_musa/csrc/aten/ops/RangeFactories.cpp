#include <ATen/AccumulateType.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/adaption.h>

#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/RangeFactories.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <mudnn.h>

namespace at {

namespace native {
DEFINE_DISPATCH(arange_start_out_stub);
REGISTER_NO_CPU_DISPATCH(arange_start_out_stub);
} // namespace native

namespace musa {

Tensor& ArangeStartOut(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  c10::musa::MUSAGuard device_guard(out.device());
  at::native::arange_start_out_stub(kMUSA, start, end, step, out);
  return out;
}
} // namespace musa
} // namespace at
