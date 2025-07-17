#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/core/Tensor.h>
#include <ATen/musa/EmptyTensor.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/musa/Resize.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <ATen/musa/MUSAApplyUtils.muh>
#include <ATen/native/musa/Loops.muh>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_efficientzerotensor_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/empty_strided_native.h>
#include <ATen/ops/eye_native.h>
#include <ATen/ops/tril_indices_native.h>
#include <ATen/ops/tril_native.h>
#include <ATen/ops/triu_indices_native.h>
#include <ATen/ops/triu_native.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>

// #include "torch_musa/csrc/aten/ops/TensorFactory.h"
namespace at::musa {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {
// To find the max integer that does not exceed the root of an int64_t variable,
// we could use a loop to test one bit at a time, which takes up to 31
// iterations. This would give the accurate result, but is relatively slow and
// is an overkill for most cases where double's precision suffice.
//
// If we directly use sqrt to calculate the root, the conversion from int64_t
// to double would lose 11 bits precision.
//
// The following solution uses sqrt directly for most cases, and would only
// special handle it if there is indeed precision loss.
__device__ inline int64_t resolve_root_int(
    int64_t b,
    int64_t cX4,
    int64_t x,
    int32_t sign) {
  int64_t bXb_cX4 = b * b - cX4;
  // potential precision loss could occur here when casting int64_t (63 bits
  // precision) to double (52 bits precision)
  double sr = ::sqrt((double)bXb_cX4);
  int64_t res = ::__double2ll_rd((-b + sign * sr) / 2);

  // have to cast double to int64_t, otherwise it would only compare up to the
  // precision of a double variable, ignoring the precision loss
  if (bXb_cX4 != (int64_t)(sr * sr)) {
    // handle precision loss by using binary search
    int64_t llsr = ::__double2ll_rd(sr);
    // Use the following math to reduce search space.
    // Suppose z is the accurate result of sqrt(bXb_cX4) without precision loss
    // let d = abs(bXb_cX4 - llsr * llsr), then we have:
    // z = sqrt(bXb_cX4) <= sqrt(llsr * llsr + d) <= llsr + sqrt(d)
    // z = sqrt(bXb_cX4) >= sqrt(llsr * llsr - d) >= llsr - sqrt(d)
    // Hence, it is sufficient to search range [llsr - sqrt(d), llsr + sqrt(d)).
    // And the true value of row would also be with in range,
    //            [res - sqrt(d), res + sqrt(d) + 1)
    // as the denominator would only reduce the precision penalty.
    int64_t diff =
        ::__double2ll_ru(::sqrt(::fabs((double)(bXb_cX4 - llsr * llsr))));
    // l never exceeds (could equal to) the target row index
    auto l = res > diff ? res - diff : 0;
    // r is always larger than the target row index
    auto r = res + diff + 1;

    // binary search for the correct answer
    x <<= 1; // the loop always compares with 2x, so do it once here
    while (l + 1 < r) {
      auto m = (l + r) >> 1;
      // for tril:
      //    b = 2f - 1, sign = 1, hence (2f + m - 1) * m / 2
      // for triu:
      //    b = -2f - 1, sign = -1, hence (2f - m + 1) * m / 2
      if (sign * (b + m) * m > x) {
        r = m;
      } else {
        l = m;
      }
    }
    res = l;
  }

  return res;
}

// f: the number of elements in the first row of the trapezoid.
// x: the index of the target coordinates ordered by row and then column.
//
// View the tril as a top trapezoid stacked on a bottom rectangle. Assume x
// corresponds to the coordinate (row, col) in the trapezoid, where the row and
// the col both start from 0, then we have:
//
//                   (f + f + row - 1) * row / 2 <= x                       [1]
//                 (f + f + row) * (row + 1) / 2  > x                       [2]
//
// Therefore, row is the maximum integer satisfying the following inequality:
//
//                       (row + 2f - 1)row <= 2x
//                  row^2 + (2f-1)row - 2x <= 0.                            [3]
//
// Based on inequality [3], we have the following coefficients for formula of
// root:
//                               a = 1
//                               b = 2f - 1
//                               c = -2x
// There are two roots, and we should use the largest integer that does not
// exceed the root on the right. Intuitively, it is because:
//  i)  the valid solution range of row is between two roots, as it is <= 0;
//  ii) as we count in more rows, the total # of elements should always
//      increase, hence so does the left-hand side row^2 + (2f-1)row - 2x.
//      Therefore, the valid range of row lies in between the nadir point and
//      the larger root on the right.
// Full proof can be derived from inequality [2]. So, we calculate the result
// coordinate as:
//
//                   row = floor((-b + sqrt(b^2 - 4c)) / 2)
//                   col = x - (f + f + row - 1) * row / 2
__device__ inline void get_coordinate_in_tril_trapezoid(
    int64_t f,
    int64_t x,
    int64_t& row,
    int64_t& col) {
  f <<= 1; // all statements use 2f, so only calculate it once here.
  auto b = f - 1;
  auto cX4 = -(x << 3); // 4 * c = 4 * (-2x) = -8x;
  row = resolve_root_int(b, cX4, x, 1);
  col = x - ((f + row - 1) * row >> 1);
}

// f: the number of elements in the first row of the bottom trapezoid.
// x: the index of the target coordinates ordered by row and then column.
//
// View the triu as a top rectangle stacked on a bottom trapezoid, where the
// trapezoid is upside down. Assume x corresponds to the coordinate (row, col)
// in the bottom trapezoid, where the row and the col start from 0, then we
// have:
//
//                   (f + f - row + 1) * row / 2 <= x                       [1]
//                 (f + f - row) * (row + 1) / 2  > x                       [2]
//
// Therefore, row is the maximum integer satisfying the following inequality:
//
//                       (-row + 2f + 1)row <= 2x
//                   row^2 - (2f+1)row + 2x >= 0.                           [3]
//
// Based on inequality [3], we have the following coefficients for formula of
// root:
//                               a = 1
//                               b = -1 - 2f
//                               c = 2x
// There are two roots, and we should use the largest integer that does not
// exceed the root on the left. Intuitively, it is because:
//  i)  the valid solution range of row is outside of the two roots, as it is <
//      > 0;
//  ii) as we count in more rows, the total # of elements should always
//      increase, hence so does the left-hand side row^2 - (2f+1)row + 2x.
//      Therefore, the valid range of row lies to the left of the smaller root
//      on the left.
// Full proof can be derived from inequality [2]. So, we calculate the result
// coordinate as:
//
//                   row = floor((-b - sqrt(b^2 - 4c)) / 2)
//                   col = x - (f + f - row + 1) * row / 2
__device__ inline void get_coordinate_in_triu_trapezoid(
    int64_t f,
    int64_t x,
    int64_t& row,
    int64_t& col) {
  f <<= 1; // all statements use 2f, so only calculate it once here.
  auto b = -1 - f;
  auto cX4 = x << 3; // 4 * c = 4 * (2x) = 8x;
  row = resolve_root_int(b, cX4, x, -1);
  col = x - ((f - row + 1) * row >> 1) + row;
}

} // namespace

template <typename scalar_t>
__global__
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_1(512)
#endif
    void tril_indices_kernel(
        scalar_t* tensor,
        int64_t row_offset,
        int64_t m_first_row,
        int64_t col,
        int64_t trapezoid_size,
        int64_t tril_size) {
  int64_t linear_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (linear_index < tril_size) {
    int64_t r, c;
    if (linear_index < trapezoid_size) {
      // the coordinate is within the top trapezoid
      get_coordinate_in_tril_trapezoid(m_first_row, linear_index, r, c);
    } else {
      // the coordinate falls in the bottom rectangle
      auto surplus = linear_index - trapezoid_size;
      // add the height of trapezoid: m_last_row (col) - m_first_row + 1
      r = surplus / col + col - m_first_row + 1;
      c = surplus % col;
    }
    r += row_offset;

    tensor[linear_index] = r;
    tensor[linear_index + tril_size] = c;
  }
}

// Some Large test cases for the fallback binary search path is disabled by
// default to speed up CI tests and to avoid OOM error. When modifying the
// implementation, please enable them in test/test_cuda.py and make sure they
// pass on your local server.
Tensor tril_indices_musa(
    int64_t row,
    int64_t col,
    int64_t offset,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  at::native::check_args(row, col, layout_opt);

  auto tril_size = at::native::get_tril_size(row, col, offset);
  auto tensor = at::musa::EmptyMUSA(
      {2, tril_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  if (tril_size > 0) {
    auto m_first_row = offset > 0 ? std::min<int64_t>(col, 1 + offset)
                                  : // upper bounded by col
        row + offset > 0; // either 0 or 1
    auto trapezoid_row_offset = std::max<int64_t>(0, -offset);
    auto rectangle_row_offset = trapezoid_row_offset + col - m_first_row + 1;
    int64_t rectangle_size = 0;
    if (rectangle_row_offset < row) {
      rectangle_size = (row - rectangle_row_offset) * col;
    }

    dim3 dim_block = musa::getApplyBlock();
    dim3 dim_grid;
    // using tril_size instead of tensor.numel(), as each thread takes care of
    // two elements in the tensor.
    TORCH_CHECK(
        musa::getApplyGrid(tril_size, dim_grid, tensor.get_device()),
        "unable to get dim grid");

    AT_DISPATCH_INDEX_TYPES(tensor.scalar_type(), "tril_indices_cuda", [&] {
      tril_indices_kernel<<<
          dim_grid,
          dim_block,
          0,
          at::musa::getCurrentMUSAStream()>>>(
          tensor.mutable_data_ptr<index_t>(),
          trapezoid_row_offset,
          m_first_row,
          col,
          tril_size - rectangle_size,
          tril_size);
      C10_MUSA_KERNEL_LAUNCH_CHECK();
    });
  }

  return tensor;
}

template <typename scalar_t>
__global__ void triu_indices_kernel(
    scalar_t* tensor,
    int64_t col_offset,
    int64_t m_first_row,
    int64_t col,
    int64_t rectangle_size,
    int64_t triu_size) {
  int64_t linear_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (linear_index < triu_size) {
    int64_t r, c;
    if (linear_index < rectangle_size) {
      // the coordinate is within the top rectangle
      r = linear_index / col;
      c = linear_index % col;
    } else {
      // the coordinate falls in the bottom trapezoid
      get_coordinate_in_triu_trapezoid(
          m_first_row, linear_index - rectangle_size, r, c);
      r += rectangle_size / col;
    }

    c += col_offset;
    tensor[linear_index] = r;
    tensor[linear_index + triu_size] = c;
  }
}

// Some Large test cases for the fallback binary search path is disabled by
// default to speed up CI tests and to avoid OOM error. When modifying the
// implementation, please enable them in test/test_cuda.py and make sure they
// pass on your local server.
Tensor triu_indices_musa(
    int64_t row,
    int64_t col,
    int64_t offset,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  at::native::check_args(row, col, layout_opt);

  auto triu_size = row * col - at::native::get_tril_size(row, col, offset - 1);
  auto tensor = at::musa::EmptyMUSA(
      {2, triu_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  if (triu_size > 0) {
    // # of triu elements in the first row
    auto m_first_row = offset > 0 ? std::max<int64_t>(col - offset, 0)
                                  : // upper bounded by col
        col;

    // size of the top rectangle
    int64_t rectangle_size = 0;
    if (offset < 0) {
      rectangle_size = std::min<int64_t>(row, -offset) * col;
    }

    dim3 dim_block = musa::getApplyBlock();
    dim3 dim_grid;

    // using triu_size instead of tensor.numel(), as each thread takes care of
    // two elements in the tensor.
    TORCH_CHECK(
        musa::getApplyGrid(triu_size, dim_grid, tensor.get_device()),
        "unable to get dim grid");

    AT_DISPATCH_INDEX_TYPES(tensor.scalar_type(), "triu_indices_cuda", [&] {
      triu_indices_kernel<<<
          dim_grid,
          dim_block,
          0,
          at::musa::getCurrentMUSAStream()>>>(
          tensor.mutable_data_ptr<index_t>(),
          std::max<int64_t>(0, offset),
          m_first_row,
          col,
          rectangle_size,
          triu_size);
      C10_MUSA_KERNEL_LAUNCH_CHECK();
    });
  }

  return tensor;
}

} // namespace at::musa
