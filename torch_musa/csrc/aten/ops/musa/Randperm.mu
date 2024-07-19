#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorFactories.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/randperm_native.h>
#endif

#include <ATen/native/musa/Randperm.muh>
#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <limits>

namespace at {
namespace musa {

namespace {

template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

} // anonymous namespace

Tensor& RandpermOutMusa(
    int64_t n,
    c10::optional<Generator> generator,
    Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  at::native::check_supported_max_int_with_precision(n, result);

  // muDNN's radix_sort_by_key only support int32 & int64 dtype of value
  bool is_supported_value_dtype =
      (result.scalar_type() == at::ScalarType::Int) ||
      (result.scalar_type() == at::ScalarType::Long);
  auto shuffled_dtype = result.element_size() <= 4 ? kInt : kLong;

  result.resize_({n});

  auto range = at::arange(n, result.options().dtype(shuffled_dtype));

  // shuffled_data points to the underlying data of the output tensor if the
  // tensor is contiguous and result.scalar_type() is Int or Long; otherwise it
  // points to a new tensor.
  Tensor shuffled;
  void* shuffled_data;
  if (result.is_contiguous() && is_supported_value_dtype) {
    shuffled = result;
    shuffled_data = result.data_ptr();
  } else {
    shuffled = at::empty(n, result.options().dtype(shuffled_dtype));
    shuffled_data = shuffled.data_ptr();
  }

  auto opt = TensorOptions().device(result.device());

  const double log_threshold_l2 = std::log(0.9) * 12;
  double nd = static_cast<double>(n);

  int bits = std::min(
      64,
      static_cast<int>(
          std::ceil(std::log2(nd - (6 * nd * nd + 1) / log_threshold_l2))));

  if (0 == n) {
    return result;
  } else if (bits <= 32) {
    // For asserting device type match of the generator and result,
    // we deligate that to the `random_` function below.

    auto keys = at::empty(result.sizes(), opt.dtype(kInt))
                    .random_(
                        std::numeric_limits<int>::min(),
                        std::numeric_limits<int>::max(),
                        generator);
    auto keys_out = at::empty_like(keys);
    auto keys_out_ptr = keys_out.data_ptr<int>();
    AT_DISPATCH_ALL_TYPES_AND(
        kHalf, result.scalar_type(), "RandpermOutMusa", [&] {
          using dtype = OpaqueType<sizeof(scalar_t)>;
          auto shuffled_data_ = reinterpret_cast<dtype*>(shuffled_data);

          muHandle& h = GetMudnnHandle();
          auto keys_mu = CreateMUTensor(keys);
          auto keys_out_mu = CreateMUTensor(keys_out);
          auto range_mu = CreateMUTensor(range);
          auto shuffled_mu = CreateMUTensor(shuffled);

          ::musa::dnn::SortByKey op;
          op.SetDim(0);
          op.SetDescending(false);
          op.SetStable(true);
          CHECK_MUDNN_STATUS(
              op.Run(
                  h,
                  keys_out_mu,
                  shuffled_mu,
                  keys_mu,
                  range_mu,
                  at::musa::InternalMemAlloc),
              "SortRunByKey");

          randperm_handle_duplicate_keys(
              keys_out_ptr, shuffled_data_, 32, n, generator);
        });
  } else {
    auto keys = at::empty(result.sizes(), opt.dtype(kLong))
                    .random_(
                        std::numeric_limits<int64_t>::min(),
                        std::numeric_limits<int64_t>::max(),
                        generator);
    auto keys_out = at::empty_like(keys);
    auto keys_out_ptr = keys_out.data_ptr<int64_t>();
    AT_DISPATCH_ALL_TYPES_AND(
        kHalf, result.scalar_type(), "RandpermOutMusa", [&] {
          using dtype = OpaqueType<sizeof(scalar_t)>;
          auto shuffled_data_ = reinterpret_cast<dtype*>(shuffled_data);

          muHandle& h = GetMudnnHandle();
          auto keys_mu = CreateMUTensor(keys);
          auto keys_out_mu = CreateMUTensor(keys_out);
          auto range_mu = CreateMUTensor(range);
          auto shuffled_mu = CreateMUTensor(shuffled);

          ::musa::dnn::SortByKey op;
          op.SetDim(0);
          op.SetDescending(false);
          op.SetStable(true);
          CHECK_MUDNN_STATUS(
              op.Run(
                  h,
                  keys_out_mu,
                  shuffled_mu,
                  keys_mu,
                  range_mu,
                  at::musa::InternalMemAlloc),
              "SortRunByKey");

          randperm_handle_duplicate_keys(
              keys_out_ptr, shuffled_data_, 64, n, generator);
        });
  }

  if (!result.is_contiguous() && !is_supported_value_dtype) {
    result.copy_(shuffled);
  }

  return result;
}

} // namespace musa
} // namespace at
