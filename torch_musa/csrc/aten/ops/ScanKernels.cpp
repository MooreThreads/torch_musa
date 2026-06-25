#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>

#include <ATen/native/ReduceOps.h>
#include <ATen/native/musa/ScanKernels.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_logcumsumexp_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

// NOTICE(ai-infra): we make this namepsace native because we don't
// want to compile generated ScanKernels.cpp from torch
namespace at::native {
namespace {

static c10::MaybeOwned<Tensor> contiguous_out_arg(const Tensor& tensor) {
  if (tensor.is_contiguous()) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
  return c10::MaybeOwned<Tensor>::owned(
      at::empty(tensor.sizes(), tensor.options()));
}
} // namespace

Tensor& _logcumsumexp_out_cuda(
    const Tensor& self,
    int64_t dim,
    Tensor& result) {
  const auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  result.resize_(self.sizes());
  if (self.dim() == 0) {
    result.copy_(self);
    return result;
  }
  if (self.numel() == 0) {
    result.zero_();
    return result;
  }

  TensorArg output_arg{result, "output", 1};
  TensorArg input_arg{self, "input", 2};
  checkAllSameGPU(__func__, {output_arg, input_arg});

  auto result_ = contiguous_out_arg(result);
  launch_logcumsumexp_cuda_kernel(*result_, self, wrap_dim);
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}

Tensor _logcumsumexp_cuda(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  return _logcumsumexp_out_cuda(self, dim, result);
}
} // namespace at::native
