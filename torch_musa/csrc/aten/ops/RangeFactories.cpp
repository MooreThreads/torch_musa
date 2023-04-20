#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace native {
namespace musa {

Tensor& ArangeStartOut(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result) {
  MUSA_TENSOR_TYPE_CHECK(result);
  double size_d =
      std::ceil((end.toDouble() - start.toDouble()) / step.toDouble());
  int64_t size = static_cast<int64_t>(size_d);
  std::vector<int64_t> shape{size};
  result.resize_(shape);
  auto out = CreateMUTensor(result);
  ::musa::dnn::Handle h;
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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("arange.out", &ArangeOut);
  m.impl("arange.start_out", &ArangeStartOut);
}

} // namespace musa
} // namespace native
} // namespace at
