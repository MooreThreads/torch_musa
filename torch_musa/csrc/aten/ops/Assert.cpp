#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorMeta.h>
#include <ATen/ops/empty.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
namespace musa {

bool IsNonzero(const Tensor& self) {
  auto n = self.numel();
  TORCH_CHECK(n == 1, "Boolean value of Tensor must be one value");

  Scalar localScalar = self.item();
  if (localScalar.isFloatingPoint()) {
    return localScalar.to<double>() != 0;
  } else if (localScalar.isComplex()) {
    return localScalar.to<c10::complex<double>>() !=
        c10::complex<double>(0.0, 0.0);
  } else if (localScalar.isIntegral(false)) {
    return localScalar.to<int64_t>() != 0;
  } else if (localScalar.isBoolean()) {
    return localScalar.to<bool>();
  }
  TORCH_INTERNAL_ASSERT(false, "Expected non-Tensor backend scalar");
}

void _AssertAsync(const Tensor& self) {
  TORCH_CHECK(
      IsNonzero(self),
      "Expected Tensor with single nonzero value, but got zero");
}

void _AssertAsyncMsg(const Tensor& self, std::string_view assert_msg) {
  TORCH_CHECK(
      IsNonzero(self), assert_msg != "" ? assert_msg : "Assertion is failed");
}

} // namespace musa

} // namespace at