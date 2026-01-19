#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/core/Tensor.h>
#include <ATen/musa/MUSAEvent.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/musa/Loops.muh>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/CachingHostAllocator.h"
#include "torch_musa/csrc/core/PeerToPeerAccess.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

#include "torch_musa/csrc/core/MUSACachingAllocator.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at::native {

void neg_kernel_musa(TensorIteratorBase& iter);
void conj_kernel_musa(TensorIteratorBase& iter);

void float16_copy_kernel_musa(TensorIteratorBase& iter) {
  gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
    return static_cast<at::Half>(value);
  });
}

void bfloat16_copy_kernel_musa(TensorIteratorBase& iter) {
  gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
    return static_cast<at::BFloat16>(value);
  });
}

void direct_copy_kernel_cuda(TensorIteratorBase& iter) {
  ScalarType dtype = iter.dtype(0);
  if (isQIntType(dtype)) {
    AT_DISPATCH_QINT_TYPES(dtype, "copy_", [&] {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    });
  } else if (
      iter.dtype(1) == kFloat && (dtype == kBFloat16 || dtype == kHalf)) {
    if (dtype == kBFloat16) {
      bfloat16_copy_kernel_musa(iter);
    } else {
      float16_copy_kernel_musa(iter);
    }
  } else if (isBitsType(dtype)) {
    TORCH_CHECK(
        dtype == iter.dtype(1),
        "copy_() does not support casting "
        "bits types to different bits types. Source dtype is ",
        iter.dtype(1),
        "target dtype is ",
        dtype);
    AT_DISPATCH_BIT_TYPES(dtype, "copy_", [&] {
      gpu_kernel_nocast(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    });
  } else {
    AT_DISPATCH_V2(
        dtype,
        "copy_",
        AT_WRAP(
            [&] { gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; }); }),
        AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
        kHalf,
        kBool,
        kBFloat16,
        kComplexHalf,
        AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
}

void neg_conj_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "neg_conj_musa", [&] {
    gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return -std::conj(x); });
  });
}
} // namespace at::native
