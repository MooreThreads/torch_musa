#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/jit_macros.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/musa/Reduce.muh>
#include "torch_musa/csrc/aten/ops/musa/Reduce.muh"

namespace at::native {

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
struct nansum_functor {
  void operator()(TensorIterator& iter) {
    at::native_musa_reduce::gpu_reduce_kernel<scalar_t, out_t>(
        iter, NanSumOps<acc_t, out_t>{});
  }
};

CONSTEXPR_EXCEPT_WIN_CUDA char nansum_name[] = "nansum";
template <typename scalar_t>
struct nansum_functor_complex {
#if AT_USE_JITERATOR()
  void operator()(TensorIterator& iter) {
    std::string func = jiterator_stringify(arg_t combine(arg_t a, scalar_t b) {
      return a + (std::isnan(b) ? arg_t{0.} : arg_t{b});
    });
    at::native_musa_reduce::
        jitted_gpu_reduce_kernel<nansum_name, scalar_t, scalar_t>(
            iter, func, 0.);
  }
#else
  void operator()(TensorIterator& iter) {
    using acc_t = at::opmath_type<scalar_t>;
    at::native_musa_reduce::gpu_reduce_kernel<scalar_t, acc_t>(
        iter, NanSumOps<acc_t, acc_t>{});
  }
#endif
};

// The function `reduce_dispatch` below dispatches to the kernel based
// on the type of `iter`. It takes care of the common logic
// for handling Half-Precision floating types.
// Otherwise the functor `op` is called to dispatch to the kernel
// of relevant type.
//
// Note: Functor `op` should take care of all the types to be supported
//       except for `at::Half` and `at::BFloat16`.
template <
    template <
        typename scalar_t,
        typename acc_t = scalar_t,
        typename out_t = scalar_t>
    typename OpFunctor,
    typename GeneralDispatcher>
static void reduce_dispatch(TensorIterator& iter, GeneralDispatcher op) {
  if (iter.dtype() == kHalf) {
    return OpFunctor<at::Half, float>{}(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return OpFunctor<at::Half, float, float>{}(iter);
  } else if (iter.dtype() == kBFloat16) {
    return OpFunctor<at::BFloat16, float>{}(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return OpFunctor<at::BFloat16, float, float>{}(iter);
  }
  op(iter);
}

static void nansum_kernel_cuda(TensorIterator& iter) {
  auto general_dispatcher = [](TensorIterator& iter) {
    auto dtype = iter.dtype();
    if (at::isComplexType(dtype)) {
      AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "nansum_cuda", [&]() {
        nansum_functor_complex<scalar_t>{}(iter);
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "nansum_cuda", [&]() {
        nansum_functor<scalar_t>{}(iter);
      });
    }
  };

  reduce_dispatch<nansum_functor>(iter, general_dispatcher);
}

REGISTER_DISPATCH(nansum_stub, &nansum_kernel_cuda);

} // namespace at::native
