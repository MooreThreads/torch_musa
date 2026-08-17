#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/SharedReduceOps.h>
#include <c10/core/Scalar.h>
#include "torch_musa/csrc/aten/ops/musa/Reduce.muh"

namespace at::native {

// noinline for the reduce functors' per-element ops: only the COMPLEX path of
// abs(=hypot)/pow is heavy, so just it goes noinline (kept out of
// gpu_reduce_kernel's unrolled accumulation loop, 345s->84s); the real path
// stays inline (no reduce-loop call cost). The reduction machinery (warp/block
// reduce) itself is not noinline-able. |x| for both real and complex; complex
// path (heavy hypot) is noinline.
template <typename acc_t, typename T>
inline C10_DEVICE acc_t ni_abs(T data) {
  return static_cast<acc_t>(std::abs(at::opmath_type<T>(data)));
}
template <typename acc_t, typename T>
MUSA_DEVICE_NOINLINE acc_t ni_abs(c10::complex<T> data) {
  return static_cast<acc_t>(std::abs(at::opmath_type<c10::complex<T>>(data)));
}

// abs only when complex (NormTwo): real -> plain cast (no abs), complex ->
// noinline abs.
template <typename acc_t, typename T>
inline C10_DEVICE acc_t ni_abs_if_complex(T data) {
  return static_cast<acc_t>(data);
}
template <typename acc_t, typename T>
MUSA_DEVICE_NOINLINE acc_t ni_abs_if_complex(c10::complex<T> data) {
  return static_cast<acc_t>(std::abs(at::opmath_type<c10::complex<T>>(data)));
}

template <typename acc_t>
MUSA_DEVICE_NOINLINE acc_t ni_norm_pow(acc_t base, acc_t expo) {
  return compat_pow(base, expo);
}

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__MUSACC__)
#define MUSA_NORM_SHFL(acc_t)                                           \
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const { \
    return WARP_SHFL_DOWN(acc, offset);                                 \
  }
#else
#define MUSA_NORM_SHFL(acc_t)
#endif

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct MusaNormOneOps {
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t) const {
    return acc + ni_abs<acc_t>(data);
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }
  inline C10_DEVICE out_t project(acc_t a) const {
    return a;
  }
  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t) {
    return acc;
  }
  MUSA_NORM_SHFL(acc_t)
};

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct MusaNormTwoOps {
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t) const {
    acc_t d = ni_abs_if_complex<acc_t>(data);
    return acc + d * d;
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }
  inline C10_DEVICE out_t project(acc_t a) const {
    return device_sqrt(a);
  }
  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t) {
    return acc;
  }
  MUSA_NORM_SHFL(acc_t)
};

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct MusaAbsMaxOps {
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t) const {
    return max_propagate_nan(acc, ni_abs<acc_t>(data));
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return max_propagate_nan(a, b);
  }
  inline C10_DEVICE out_t project(acc_t a) const {
    return a;
  }
  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t) {
    return acc;
  }
  MUSA_NORM_SHFL(acc_t)
};

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct MusaAbsMinOps {
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t) const {
    return min_propagate_nan(acc, ni_abs<acc_t>(data));
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return min_propagate_nan(a, b);
  }
  inline C10_DEVICE out_t project(acc_t a) const {
    return a;
  }
  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t) {
    return acc;
  }
  MUSA_NORM_SHFL(acc_t)
};

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct MusaNormOps {
  acc_t norm_;
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t) const {
    return acc + ni_norm_pow<acc_t>(ni_abs<acc_t>(data), norm_);
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }
  inline C10_DEVICE out_t project(acc_t a) const {
    return ni_norm_pow<acc_t>(a, static_cast<acc_t>(1.0) / norm_);
  }
  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t) {
    return acc;
  }
  MUSA_NORM_SHFL(acc_t)
  MusaNormOps(acc_t norm) : norm_(norm) {}
};

// This reduction accumulates results as the type `acc_t`. By default, when
// `scalar_t` is complex, `acc_t` is the downgraded real number type.
// Otherwise, `acc_t` and `scalar_t` are the same type.
template <
    typename scalar_t,
    typename acc_t = typename scalar_value_type<scalar_t>::type,
    typename out_t = typename scalar_value_type<scalar_t>::type>
void norm_kernel_musa_impl(TensorIterator& iter, double p) {
  if (p == static_cast<double>(0)) {
    at::native_musa_reduce::gpu_reduce_kernel<scalar_t, out_t>(
        iter, NormZeroOps<scalar_t, acc_t>(), 0);
  } else if (p == static_cast<double>(1)) {
    at::native_musa_reduce::gpu_reduce_kernel<scalar_t, out_t>(
        iter, MusaNormOneOps<scalar_t, acc_t>(), 0);
  } else if (p == static_cast<double>(2)) {
    at::native_musa_reduce::gpu_reduce_kernel<scalar_t, out_t>(
        iter, MusaNormTwoOps<scalar_t, acc_t>(), 0);
  } else if (p == static_cast<double>(INFINITY)) {
    at::native_musa_reduce::gpu_reduce_kernel<scalar_t, out_t>(
        iter, MusaAbsMaxOps<scalar_t, acc_t>(), 0);
  } else if (p == static_cast<double>(-INFINITY)) {
    at::native_musa_reduce::gpu_reduce_kernel<scalar_t, out_t>(
        iter,
        MusaAbsMinOps<scalar_t, acc_t>(),
        std::numeric_limits<acc_t>::infinity());
  } else {
    at::native_musa_reduce::gpu_reduce_kernel<scalar_t, out_t>(
        iter, MusaNormOps<scalar_t, acc_t>{acc_t(p)}, 0);
  }
}

void norm_launch_kernel(TensorIterator& iter, double ord) {
  if (iter.dtype(0) == kHalf) {
    return norm_kernel_musa_impl<at::Half, float>(iter, ord);
  } else if (iter.input_dtype() == kHalf && iter.dtype(0) == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_musa_impl<at::Half, float, float>(iter, ord);
  } else if (iter.dtype(0) == kBFloat16) {
    return norm_kernel_musa_impl<at::BFloat16, float>(iter, ord);
  } else if (iter.input_dtype() == kBFloat16 && iter.dtype(0) == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_musa_impl<at::BFloat16, float, float>(iter, ord);
  } else if (iter.input_dtype() == kDouble && at::musa::getMUSAArch() <= 210) {
    // Due to a bug in the mcc of QY1,this operator does not support
    // double precision inputs on QY1.
    TORCH_CHECK(
        false, "norm_launch_kernel does not support input of double type.");
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.input_dtype(), "norm_musa", [&] {
    norm_kernel_musa_impl<scalar_t>(iter, ord);
  });
}

} // namespace at::native
