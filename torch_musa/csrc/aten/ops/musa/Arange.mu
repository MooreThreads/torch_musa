#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/RangeUtils.h>
#include <c10/util/MaybeOwned.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

#include "torch_musa/csrc/aten/musa/MUSAContextLight.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"
// clang-format off
// #include "torch_musa/csrc/aten/ops/musa/elemwise/MemoryAccess.muh"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
// clang-format on

#include <algorithm>
#include <cmath>
#include <limits>

namespace at::musa {

namespace {

template <typename scalar_t, typename accscalar_t>
__global__ void arange_kernel(
    scalar_t* out_ptr,
    accscalar_t start,
    accscalar_t step,
    int64_t numel) {
  constexpr int vec_size = 128 / (sizeof(scalar_t) * 8);
  // using vec_t = aligned_vector<scalar_t, vec_size>;
  using vec_t = VecType<scalar_t, 128>;

  const auto stride = (int64_t)(blockDim.x) * gridDim.x * vec_size;
  auto idx = (threadIdx.x + (int64_t)(blockIdx.x) * blockDim.x) * vec_size;

  accscalar_t base;
  while (idx + vec_size <= numel) {
    vec_t vec_o;
    base = start + (accscalar_t)(idx)*step;
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      // vec_o.val[i] = (scalar_t)(base);
      vec_o.val_.elem[i] = (scalar_t)(base);
      base += step;
    }
    // *(vec_t*)(out_ptr + idx) = vec_o;
    vec_t::store(out_ptr, idx, vec_o);
    idx += stride;
  }

  base = start + (accscalar_t)(idx)*step;
  while (idx < numel) {
    out_ptr[idx] = (scalar_t)(base);
    base += step;
    ++idx;
  }
}

} // namespace

Tensor& ArangeStartOut(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  const auto* dev_prop = getCurrentDeviceProperties();
  const int mp_num = dev_prop->multiProcessorCount;
  const int major = dev_prop->major;
  const int minor = dev_prop->minor;
  const musaStream_t stream = getCurrentMUSAStream();
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      out.scalar_type(),
      "arange_musa",
      [&, mp_num, major, minor, stream]() {
        using accscalar_t = at::acc_type<scalar_t, true>;
        auto xstart = start.to<accscalar_t>();
        auto xend = end.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        native::arange_check_bounds(start, end, step);

        double size_d;
        if constexpr (std::is_same_v<scalar_t, int64_t>) {
          int64_t sgn = (xstep > 0) - (xstep < 0);
          size_d = std::ceil((xend - xstart + xstep - sgn) / xstep);
        } else {
          size_d = std::ceil(
              static_cast<double>(end.to<double>() - start.to<double>()) /
              step.to<double>());
        }

        TORCH_CHECK(
            size_d >= 0 &&
                size_d <=
                    static_cast<double>(std::numeric_limits<int64_t>::max()),
            "invalid size, possible overflow?");
        const auto size = static_cast<int64_t>(size_d);
        int64_t numel = out.numel();

        if (numel != size) {
          if (numel > 0) {
            TORCH_WARN(
                "The number of elements in the out tensor of shape ",
                out.sizes(),
                " is ",
                numel,
                " which does not match the computed number of elements ",
                size,
                ". Note that this may occur as a result of rounding error. "
                "The out tensor will be resized to a tensor of shape (",
                size,
                ",).");
          }
          out.resize_({size});
          numel = size;
        }

        if (numel == 0) {
          return;
        }

        const bool is_contig = out.is_contiguous();
        auto r = is_contig ? c10::MaybeOwned<Tensor>::borrowed(out)
                           : c10::MaybeOwned<Tensor>::owned(at::empty_like(
                                 out, LEGACY_CONTIGUOUS_MEMORY_FORMAT));

        constexpr int64_t vec_size = 128 / (sizeof(scalar_t) * 8);
        constexpr int elemsize = sizeof(scalar_t);
        int64_t threads = elemsize < 4 ? 1024 : (elemsize < 8 ? 512 : 256);
        int64_t blocks = ceil_div(numel, vec_size * threads);

        if (major < 2 || (major == 2 && minor <= 1)) {
          blocks = std::min(blocks, static_cast<int64_t>(mp_num));
        } else {
          while (blocks < mp_num && threads > 256) {
            blocks <<= 1;
            threads >>= 1;
          }
        }
        blocks = std::max<int64_t>(1, blocks);

        arange_kernel<scalar_t, accscalar_t><<<blocks, threads, 0, stream>>>(
            r->data_ptr<scalar_t>(), xstart, xstep, numel);
        C10_MUSA_KERNEL_LAUNCH_CHECK();

        if (!is_contig) {
          out.copy_(*r);
        }
      });

  return out;
}

} // namespace at::musa
