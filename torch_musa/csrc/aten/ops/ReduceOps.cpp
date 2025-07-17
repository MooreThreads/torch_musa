#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/musa/ReduceOps.h>

#include <ATen/Context.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/NamedTensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/amax_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/aminmax_native.h>
#include <ATen/ops/full.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/kthvalue_native.h>
#include <ATen/ops/median_native.h>
#include <ATen/ops/nanmedian_native.h>
#include <ATen/ops/where.h>
#endif

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/musa/ReduceOps.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/aminmax_native.h>
#include <torch/library.h>

#include <tuple>

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
namespace native {
namespace {

void NormKernel(TensorIterator& iter, const Scalar& val) {
  double p;
  if (val.isIntegral(false)) {
    p = val.to<int64_t>();
  } else if (val.isFloatingPoint()) {
    p = val.to<double>();
  } else {
    TORCH_CHECK(
        false, "norm_launch_kernel expects norm to be integer or float");
  }
  if (iter.numel() == 0) {
    iter.output().fill_((p < 0) ? INFINITY : 0);
    return;
  }
  norm_launch_kernel(iter, p);

  if (isComplexType(iter.output().scalar_type())) {
    at::imag(iter.output()).zero_();
  }
}

void AMinMaxKernel(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& min_result,
    Tensor& max_result) {
  at::TensorIterator iter = make_reduction(
      "aminmax_musa",
      min_result,
      max_result,
      self,
      dim,
      keepdim,
      self.scalar_type());
  aminmax_launch_kernel(iter);
}

void AMinMaxAllReduceKernel(
    const Tensor& input,
    Tensor& min_result,
    Tensor& max_result) {
  auto dtype = input.scalar_type();
  auto iter = make_reduction(
      "aminmax_musa",
      min_result,
      max_result,
      input,
      IntArrayRef{},
      false,
      dtype);
  TORCH_CHECK(
      iter.numel() > 0, "min_max on a tensor with no elements is not defined.");
  aminmax_allreduce_launch_kernel(iter);
}

} // namespace

REGISTER_MUSA_DISPATCH(norm_stub, &NormKernel);
REGISTER_MUSA_DISPATCH(aminmax_stub, &AMinMaxKernel);
REGISTER_MUSA_DISPATCH(aminmax_allreduce_stub, &AMinMaxAllReduceKernel);

} // namespace native

namespace musa {

using Mode = ::musa::dnn::Reduce::Mode;

at::Tensor& ReduceOp(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    at::Tensor& out,
    Mode mode) {
  if (C10_UNLIKELY(self.numel() == 0)) {
    return out;
  }
  c10::musa::MUSAGuard device_guard(self.device());
  DimVector dims_vec(dim);
  maybe_wrap_dims(dims_vec, self.dim());
  namedinference::propagate_names_for_reduction(out, self, dims_vec, keepdim);

  at::Tensor input = FormatContiguous(self, MemoryFormat::Contiguous);
  auto in_m = CreateMUTensor(input);
  auto out_m = CreateMUTensor(out);

  std::vector<int> dims(dims_vec.data(), dims_vec.data() + dims_vec.size());
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Reduce r;
  CHECK_MUDNN_STATUS(r.SetMode(mode), "SetMode");
  CHECK_MUDNN_STATUS(r.SetDim(dims.size(), dims.data()), "SetDim");
  CHECK_MUDNN_STATUS(r.Run(h, out_m, in_m, InternalMemAlloc), "Run");

  return out;
}

at::Tensor& AMinOut(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    at::Tensor& out) {
  Mode mode = ::musa::dnn::Reduce::Mode::MIN;
  return ReduceOp(self, dim, keepdim, out, mode);
}

at::Tensor& AMaxOut(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    at::Tensor& out) {
  Mode mode = ::musa::dnn::Reduce::Mode::MAX;
  return ReduceOp(self, dim, keepdim, out, mode);
}

std::tuple<at::Tensor&, at::Tensor&> AMinMaxOut(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    at::Tensor& min,
    at::Tensor& max) {
  if (C10_UNLIKELY(self.numel() == 0)) {
    return std::forward_as_tuple(min, max);
  }
  c10::musa::MUSAGuard device_guard(self.device());
  at::Tensor input = FormatContiguous(self, MemoryFormat::Contiguous);

  auto in_m = CreateMUTensor(input);
  auto min_m = CreateMUTensor(min);
  auto max_m = CreateMUTensor(max);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Reduce r_min;
  ::musa::dnn::Reduce r_max;
  CHECK_MUDNN_STATUS(
      r_min.SetMode(::musa::dnn::Reduce::Mode::MIN), "SetMinMode");
  CHECK_MUDNN_STATUS(
      r_max.SetMode(::musa::dnn::Reduce::Mode::MAX), "SetMaxMode");
  if (dim.has_value()) {
    IntArrayRef dims(dim.value());
    DimVector dims_vec(dims);
    maybe_wrap_dims(dims_vec, self.dim());
    namedinference::propagate_names_for_reduction(min, self, dims_vec, keepdim);
    namedinference::propagate_names_for_reduction(max, self, dims_vec, keepdim);
    const int dim_m = dims_vec[0];
    CHECK_MUDNN_STATUS(r_min.SetDim({dim_m}), "SetMinDim");
    CHECK_MUDNN_STATUS(r_max.SetDim({dim_m}), "SetMaxDim");
  }
  CHECK_MUDNN_STATUS(r_min.Run(h, min_m, in_m, InternalMemAlloc), "RunMin");
  CHECK_MUDNN_STATUS(r_max.Run(h, max_m, in_m, InternalMemAlloc), "RunMax");

  return std::forward_as_tuple(min, max);
}

std::tuple<at::Tensor, at::Tensor> AMinMax(
    const at::Tensor& self,
    c10::optional<int64_t> dim_opt,
    bool keepdim) {
  DimVector shape;
  if (dim_opt.has_value()) {
    auto dim = maybe_wrap_dim(dim_opt.value(), self.ndimension());
    at::native::zero_numel_check_dims(self, dim, "aminmax");
    shape = at::meta::get_reduction_shape(self, dim, keepdim);
  } else {
    TORCH_CHECK(
        self.numel() > 0,
        "aminmax(): cannot compute aminmax over an empty dimension as the "
        "operation has no identity.");
    if (keepdim) {
      shape = DimVector(self.ndimension(), 1);
    }
  }
  at::Tensor min = at::empty(shape, self.options());
  at::Tensor max = at::empty(shape, self.options());
  return AMinMaxOut(self, dim_opt, keepdim, min, max);
}

std::tuple<at::Tensor, at::Tensor> AMinMaxAll(const at::Tensor& self) {
  TORCH_WARN_ONCE(
      "_aminmax is deprecated as of PyTorch 1.11 and will be removed in a future release. Use aminmax instead."
      " This warning will only appear once per process.");
  return at::aminmax(self);
}

std::tuple<Tensor, Tensor> AMinMax_(
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  TORCH_WARN_ONCE(
      "_aminmax is deprecated as of PyTorch 1.11 and will be removed in a future release. Use aminmax instead."
      " This warning will only appear once per process.");
  return at::aminmax(self, dim, keepdim);
}
} // namespace musa
} // namespace at
