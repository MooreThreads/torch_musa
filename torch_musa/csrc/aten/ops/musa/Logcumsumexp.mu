#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/TensorBase.h>

#include <ATen/native/musa/ScanKernels.h>
#include <ATen/native/musa/ScanUtils.muh>

#include <cmath>
#include <limits>

namespace at::native {

template <typename scalar_t>
__host__ __device__ scalar_t
_log_add_exp_helper(const scalar_t& x, const scalar_t& y) {
  // Reference :
  // https://www.tensorflow.org/api_docs/python/tf/math/cumulative_logsumexp
  // Using the original expression: `at::_isnan(y) ? y : std::min(x, y)` causes
  // an error in ROCM
  auto isnan_x = at::_isnan(x);
  auto isnan_y = at::_isnan(y);
  scalar_t min = isnan_y ? y : (isnan_x ? x : std::min(x, y));
  scalar_t max = isnan_y ? y : (isnan_x ? x : std::max(x, y));
  if (min != max || ::isfinite(min)) {
    // nan will be propagated here
    return ::log1p(std::exp(min - max)) + max;
  } else {
    // special case to correctly handle infinite cases
    return x;
  }
}

void launch_logcumsumexp_cuda_kernel(
    const TensorBase& result,
    const TensorBase& self,
    int64_t dim) {
  // AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16,
  AT_DISPATCH_FLOATING_TYPES_AND(
      ScalarType::Half, self.scalar_type(), "logcumsumexp_musa", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        scalar_t init = -std::numeric_limits<scalar_t>::infinity();
        auto log_add_exp = [] C10_HOST_DEVICE(
                               const scalar_t x_,
                               const scalar_t y_) -> scalar_t {
          const opmath_t x{x_}, y{y_};
          return _log_add_exp_helper(x, y);
        };
        scan_dim<scalar_t>(self, result, dim, init, log_add_exp);
      });
}

} // namespace at::native
