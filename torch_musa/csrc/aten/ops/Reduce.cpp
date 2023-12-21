#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/ops/max.h>
#include <torch/library.h>
#include <sstream>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cummax_helper.h>
#include <ATen/ops/_cummax_helper_native.h>
#include <ATen/ops/_cummin_helper.h>
#include <ATen/ops/_cummin_helper_native.h>
#include <ATen/ops/_logcumsumexp.h>
#include <ATen/ops/_logcumsumexp_native.h>
#include <ATen/ops/_sparse_sum.h>
#include <ATen/ops/_sparse_sum_native.h>
#include <ATen/ops/add.h>
#include <ATen/ops/all_meta.h>
#include <ATen/ops/all_native.h>
#include <ATen/ops/amax.h>
#include <ATen/ops/amax_meta.h>
#include <ATen/ops/amax_native.h>
#include <ATen/ops/amin_meta.h>
#include <ATen/ops/amin_native.h>
#include <ATen/ops/aminmax_meta.h>
#include <ATen/ops/aminmax_native.h>
#include <ATen/ops/any_meta.h>
#include <ATen/ops/any_native.h>
#include <ATen/ops/argmax_meta.h>
#include <ATen/ops/argmax_native.h>
#include <ATen/ops/argmin_meta.h>
#include <ATen/ops/argmin_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/cummax.h>
#include <ATen/ops/cummax_native.h>
#include <ATen/ops/cummaxmin_backward_native.h>
#include <ATen/ops/cummin.h>
#include <ATen/ops/cummin_native.h>
#include <ATen/ops/cumprod.h>
#include <ATen/ops/cumprod_backward_native.h>
#include <ATen/ops/cumprod_meta.h>
#include <ATen/ops/cumprod_native.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/cumsum_meta.h>
#include <ATen/ops/cumsum_musa_dispatch.h>
#include <ATen/ops/cumsum_native.h>
#include <ATen/ops/diff_native.h>
#include <ATen/ops/dist_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/equal_native.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/gradient_native.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/isnan_native.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/logcumsumexp.h>
#include <ATen/ops/logcumsumexp_native.h>
#include <ATen/ops/logical_xor.h>
#include <ATen/ops/logsumexp.h>
#include <ATen/ops/logsumexp_native.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mean_meta.h>
#include <ATen/ops/mean_native.h>
#include <ATen/ops/min.h>
#include <ATen/ops/nanmean_native.h>
#include <ATen/ops/nansum.h>
#include <ATen/ops/nansum_native.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/native_norm.h>
#include <ATen/ops/norm.h>
#include <ATen/ops/norm_meta.h>
#include <ATen/ops/norm_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/prod.h>
#include <ATen/ops/prod_meta.h>
#include <ATen/ops/prod_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/special_logsumexp_native.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/std.h>
#include <ATen/ops/std_mean.h>
#include <ATen/ops/std_mean_native.h>
#include <ATen/ops/std_native.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/sum_meta.h>
#include <ATen/ops/sum_native.h>
#include <ATen/ops/trace_native.h>
#include <ATen/ops/value_selecting_reduction_backward_native.h>
#include <ATen/ops/var.h>
#include <ATen/ops/var_mean.h>
#include <ATen/ops/var_mean_native.h>
#include <ATen/ops/var_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

#include "torch_musa/csrc/aten/ops/Reduce.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/musa_lazy_init.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

#include <mudnn.h>

namespace at {
namespace native { // this namespace is used to define logsumexp stub only

DEFINE_DISPATCH(logsumexp_stub);
REGISTER_NO_CPU_DISPATCH(logsumexp_stub);

Tensor& LogSumExpOutImpl(Tensor& result, const Tensor& self, IntArrayRef dims) {
  // For multi-dim reduction, we iteratively run musa kernel
  // TODO(@fan.mo): optimize and speedup
  if (dims.size() > 1) {
    Tensor cur_self = self;
    Tensor cur_result;
    for (int i = 0; i < dims.size(); ++i) {
      DimVector cur_dim({dims[i]});
      maybe_wrap_dims(cur_dim, self.dim());
      auto cur_shape =
          at::meta::get_reduction_shape(cur_self, cur_dim, /*keepdim=*/true);
      cur_result = at::empty(cur_shape, result.options());
      logsumexp_stub(kMUSA, cur_result, cur_self, cur_dim[0]);
      cur_self = cur_result;
    }
    result.copy_(cur_self);
  } else {
    logsumexp_stub(kMUSA, result, self, dims[0]);
  }
  return result;
}
} // namespace native

namespace musa {

// Copy from ReduceOps.cpp
inline ScalarType musa_get_dtype_from_self(
    const Tensor& self,
    const optional<ScalarType>& dtype,
    bool promote_integers) {
  if (dtype.has_value()) {
    return dtype.value();
  }
  ScalarType src_type = self.scalar_type();
  if (promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
    return kLong;
  }
  return src_type;
}

// Copy from ReduceOps.cpp
ScalarType musa_infer_dtype_from_optional(
    const Tensor& self,
    const optional<ScalarType>& opt_dtype,
    const Tensor& result) {
  // 'opt_dtype' has the priority for both cases.
  if (result.defined()) {
    // Otherwise, get the result type, if defined.
    return opt_dtype.value_or(result.scalar_type());
  } else {
    // Last case is to get the self type.
    // If the self type is an integer, we promote it to kLong.
    return musa_get_dtype_from_self(self, opt_dtype, true);
  }
}

void ReduceCall(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    ::musa::dnn::Reduce::Mode m,
    const c10::optional<at::Scalar>& p = c10::nullopt,
    const bool is_norm = false) {
  c10::musa::MUSAGuard device_guard(self.device());
  auto input = Contiguous(self);
  auto out = CreateMUTensor(output);
  auto in = CreateMUTensor(input);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Reduce r;
  CHECK_MUDNN_STATUS(r.SetMode(m), "SetMode");
  // That input is scalar, but dim = [0] is allowed in PyTorch, in which case
  // we need pass an empty 'dim' paramter to Reduce in muDNN.
  if (self.dim() == 0 && self.numel() == 1) {
    CHECK_MUDNN_STATUS(r.SetDim({}), "SetDim");
  } else {
    std::vector<int> dim_int(dim.begin(), dim.end());
    CHECK_MUDNN_STATUS(r.SetDim(dim_int.size(), dim_int.data()), "SetDim");
  }
  // set order parameter for norm op
  if (is_norm) {
    float p_value = 2.0f;
    if (p.has_value()) {
      auto val = p.value();
      if (val.isIntegral(false)) {
        p_value = static_cast<float>(val.to<int64_t>());
      } else if (val.isFloatingPoint()) {
        p_value = static_cast<float>(val.to<double>());
      } else {
        TORCH_CHECK(
            false, "norm_kernel_musa_impl expects norm to be integer or float");
      }
    }
    CHECK_MUDNN_STATUS(r.SetNormOrd(p_value), "SetNormOrd");
  }

  CHECK_MUDNN_STATUS(r.Run(h, out, in, InternalMemAlloc), "Run");
}

Tensor Reduction(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> out_dtype,
    ::musa::dnn::Reduce::Mode m,
    const c10::optional<at::Scalar>& p = c10::nullopt,
    const bool is_norm = false) {
  c10::musa::MUSAGuard device_guard(self.device());
  out_dtype = musa_infer_dtype_from_optional(self, out_dtype, Tensor());
  DimVector dims_vec(dim);
  maybe_wrap_dims(dims_vec, self.dim());
  auto shape = at::meta::get_reduction_shape(self, dims_vec, keepdim);

  Tensor output = at::empty(shape, self.options().dtype(out_dtype));
  namedinference::propagate_names_for_reduction(
      output, self, dims_vec, keepdim);

  if (self.numel() == 0) {
    output.zero_();
  } else {
    ReduceCall(output, self, dims_vec, m, p, is_norm);
  }
  return output;
}

#define REDUCE_OPERATOR(op_name)
Tensor Mean(const Tensor& self, c10::optional<ScalarType> dtype) {
  return Reduction(
      self, IntArrayRef{}, false, dtype, ::musa::dnn::Reduce::Mode::MEAN);
}

Tensor MeanDim(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  return Reduction(
      self, dim.value(), keepdim, dtype, ::musa::dnn::Reduce::Mode::MEAN);
}

Tensor& MeanOut(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype,
    Tensor& output) {
  ReduceCall(output, self, dim.value(), ::musa::dnn::Reduce::Mode::MEAN);
  return output;
}

Tensor MeanNamesDim(
    const Tensor& self,
    DimnameList dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  return Reduction(
      self,
      dimnames_to_positions(self, dim),
      keepdim,
      dtype,
      ::musa::dnn::Reduce::Mode::MEAN);
}

Tensor& MeanNamesDimOut(
    const Tensor& self,
    DimnameList dim,
    bool keepdim,
    c10::optional<ScalarType> dtype,
    Tensor& output) {
  ReduceCall(
      output,
      self,
      dimnames_to_positions(self, dim),
      ::musa::dnn::Reduce::Mode::MEAN);
  return output;
}

Tensor Sum(const Tensor& self, c10::optional<ScalarType> dtype) {
  return Reduction(
      self, IntArrayRef{}, false, dtype, ::musa::dnn::Reduce::Mode::ADD);
}

Tensor& SumIntListOut(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype,
    Tensor& output) {
  c10::musa::MUSAGuard device_guard(self.device());
  DimVector dims_vec(dim.value());
  maybe_wrap_dims(dims_vec, self.dim());
  auto shape = at::meta::get_reduction_shape(self, dims_vec, keepdim);
  output.resize_(shape);
  ReduceCall(output, self, dim.value(), ::musa::dnn::Reduce::Mode::ADD);
  return output;
}

Tensor SumDimnameList(
    const Tensor& self,
    DimnameList dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  return Reduction(
      self,
      dimnames_to_positions(self, dim),
      keepdim,
      dtype,
      ::musa::dnn::Reduce::Mode::ADD);
}

Tensor& SumDimnameListOut(
    const Tensor& self,
    DimnameList dim,
    bool keepdim,
    c10::optional<ScalarType> dtype,
    Tensor& output) {
  UNUSED(dtype);
  UNUSED(keepdim);
  ReduceCall(
      output,
      self,
      dimnames_to_positions(self, dim),
      ::musa::dnn::Reduce::Mode::ADD);
  return output;
}

Tensor SumIntList(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return Reduction(
      self, dim.value(), keepdim, opt_dtype, ::musa::dnn::Reduce::Mode::ADD);
}

Tensor Prod(const Tensor& self, c10::optional<ScalarType> dtype) {
  return Reduction(
      self, IntArrayRef{}, false, dtype, ::musa::dnn::Reduce::Mode::PROD);
}

Tensor& ProdIntOut(
    const Tensor& self,
    long dim,
    bool keepdim,
    c10::optional<ScalarType> dtype,
    Tensor& output) {
  ReduceCall(output, self, dim, ::musa::dnn::Reduce::Mode::PROD);
  return output;
}

Tensor& NormDtypeOut(
    const Tensor& self,
    const c10::optional<at::Scalar>& p,
    at::IntArrayRef dim,
    bool keepdim,
    at::ScalarType dtype,
    at::Tensor& out) {
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Half,
      "Dtype of input tensor of Norm.out only support Float32/Half, ",
      "but now it is ",
      self.scalar_type());
  TORCH_CHECK(
      dtype == at::ScalarType::Float || dtype == at::ScalarType::Half,
      "Dtype of input tensor of Norm.out only support Float32/Half, ",
      "but now it is ",
      self.scalar_type());
  auto out_dtype = out.scalar_type();

  // special case for type promotion in mixed precision, improves computational
  // efficiency.
  const bool gpu_lowp_to_f32 =
      (self.scalar_type() == kHalf || self.scalar_type() == kBFloat16) &&
      out_dtype == kFloat;

  out = Reduction(
      self,
      dim,
      keepdim,
      gpu_lowp_to_f32 ? self.scalar_type() : out_dtype,
      ::musa::dnn::Reduce::Mode::NORM,
      p,
      true);
  return out;
}

Tensor& NormOut(
    const Tensor& self,
    const c10::optional<at::Scalar>& p,
    at::IntArrayRef dim,
    bool keepdim,
    Tensor& out) {
  out = NormDtypeOut(self, p, dim, keepdim, self.scalar_type(), out);
  return out;
}

Tensor CumsumCall(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype_opt,
    Tensor& out) {
  UNUSED(dtype_opt);
  c10::musa::MUSAGuard device_guard(self.device());
  muTensor self_mt = CreateMUTensor(self);
  muTensor out_mt = CreateMUTensor(out);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Cumsum csop;
  CHECK_MUDNN_STATUS(csop.SetDim(dim), "SetDim");
  CHECK_MUDNN_STATUS(csop.Run(h, out_mt, self_mt, InternalMemAlloc), "Run");
  return out;
}

Tensor Cumsum(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype_opt) {
  Tensor self_ = self.contiguous();
  Tensor out = at::empty_like(self);
  if (self.dtype() == at::kBool) {
    out = out.to(at::kLong);
  }
  return CumsumCall(self_, dim, dtype_opt, out);
}

Tensor& Cumsum_(
    Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype_opt) {
  Tensor self_ = self.contiguous();
  if (self.dtype() == at::kBool) {
    Tensor out = at::empty_like(self, at::kLong, at::MemoryFormat::Contiguous);
    CumsumCall(self_, dim, dtype_opt, out);
    self.copy_(out);
    return self;
  } else {
    auto out = self;
    self = CumsumCall(self_, dim, dtype_opt, out);
    return self;
  }
}

Tensor& Cumsum_Out(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype_opt,
    Tensor& out) {
  Tensor self_ = self.contiguous();
  if (self.dtype() == at::kBool) {
    Tensor out_ = at::empty_like(self, at::kLong, at::MemoryFormat::Contiguous);
    CumsumCall(self_, dim, dtype_opt, out_);
    out.copy_(out_);
    return out;
  } else {
    CumsumCall(self_, dim, dtype_opt, out);
    return out;
  }
}

// same like at::cuda::cumsum_out
TORCH_API at::Tensor& cumsum_out(
    at::Tensor& out,
    const at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype) {
  return at::musa::Cumsum_Out(self, dim, dtype, out);
}

namespace {

struct StructuredMusaAny final : public at::native::structured_any_all_out {
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }

  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
};

at::Tensor WrapperMusaAny(const at::Tensor& self) {
  StructuredMusaAny op;
  op.meta(self);
  op.impl(self, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

struct StructuredMusaAnyOut final : public at::native::structured_any_all_out {
  StructuredMusaAnyOut(Tensor& out0) : outputs_{std::ref(out0)} {}

  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
    if (C10_UNLIKELY(maybe_proxy.has_value())) {
      proxy_outputs_[output_idx] =
          c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());
    }
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }

  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
};

at::Tensor& WrapperMusaAnyOut(const at::Tensor& self, at::Tensor& out) {
  StructuredMusaAnyOut op(out);
  op.meta(self);
  op.impl(self, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value()) {
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  }
  return out;
}

} // anonymous namespace

Tensor Any(const Tensor& self) {
  if (self.scalar_type() != ScalarType::Bool) {
    return WrapperMusaAny(self);
  }
  return Reduction(
      self,
      IntArrayRef{},
      false,
      self.scalar_type(),
      ::musa::dnn::Reduce::Mode::OR);
}

Tensor& AnyOut(const Tensor& self, Tensor& out) {
  if (self.scalar_type() != ScalarType::Bool) {
    return WrapperMusaAnyOut(self, out);
  }
  IntArrayRef dims = {};
  ReduceCall(out, self, dims, ::musa::dnn::Reduce::Mode::OR);
  return out;
}

std::string concatenate(
    const std::string& str,
    ScalarType scalaType,
    const Tensor& self) {
  std::ostringstream oss;
  oss << str << scalaType << ": " << self;
  return oss.str();
}

Tensor AnyDim(const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(
      self.scalar_type() == ScalarType::Bool || self.item<int>() == 0 ||
          self.item<int>() == 1,
      concatenate(
          "Now only support bool type or 0/1 value, but got ",
          self.scalar_type(),
          self));
  IntArrayRef dims(dim);
  return Reduction(
      self, {dim}, keepdim, self.scalar_type(), ::musa::dnn::Reduce::Mode::OR);
}

Tensor& AnyDimOut(const Tensor& self, int64_t dim, bool keepdim, Tensor& out) {
  UNUSED(keepdim);
  TORCH_CHECK(
      self.scalar_type() == ScalarType::Bool || self.item<int>() == 0 ||
          self.item<int>() == 1,
      concatenate(
          "Now only support bool type or 0/1 value, but got ",
          self.scalar_type(),
          self));
  IntArrayRef dims(dim);
  ReduceCall(out, self, dims, ::musa::dnn::Reduce::Mode::OR);
  return out;
}

void ReduceIndicesCall(
    Tensor& output,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    ::musa::dnn::Reduce::Mode m) {
  TORCH_CHECK(
      self.scalar_type() == output.scalar_type(),
      "scalar_type of in&out must be the same, bug got: ",
      self.scalar_type(),
      " and: ",
      output.scalar_type());
  TORCH_CHECK(indices.scalar_type() == kLong, "Only support int64 indices now");

  c10::musa::MUSAGuard device(self.device());

  auto input = self.contiguous();

  auto out = CreateMUTensor(output);
  auto ids = CreateMUTensor(indices);
  auto in = CreateMUTensor(input);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Reduce r;
  CHECK_MUDNN_STATUS(r.SetMode(m), "SetMode");
  int dim_int = dim;
  CHECK_MUDNN_STATUS(r.SetDim({dim_int}), "SetDim");
  CHECK_MUDNN_STATUS(
      r.RunWithIndices(h, out, ids, in, InternalMemAlloc), "RunWithIndices");
}

std::tuple<Tensor, Tensor> ReductionIndices(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    ::musa::dnn::Reduce::Mode m) {
  dim = maybe_wrap_dim(dim, self.dim());

  IntArrayRef dims(dim);
  DimVector dims_vec(dims);
  maybe_wrap_dims(dims_vec, self.dim());
  auto shape = at::meta::get_reduction_shape(self, dims_vec, keepdim);

  auto out_dtype = self.scalar_type();
  Tensor output = at::empty(shape, self.options().dtype(out_dtype));
  Tensor indices = at::empty(shape, self.options().dtype(kLong));
  namedinference::propagate_names_for_reduction(
      output, self, dims_vec, keepdim);
  namedinference::propagate_names_for_reduction(
      indices, self, dims_vec, keepdim);

  ReduceIndicesCall(output, indices, self, dim, m);
  return std::make_tuple(output, indices);
}

Tensor MaxAllCall(const Tensor& self, ::musa::dnn::Reduce::Mode m) {
  auto out_dtype = self.scalar_type();
  // torch.max call reudce_all according to out.dim
  Tensor output = at::empty({}, self.options().dtype(out_dtype));
  DimVector dims_vec(0);
  if (self.numel() == 0) {
    output.zero_();
  } else {
    ReduceCall(output, self, dims_vec, m);
  }
  return output;
}

Tensor MaxAll(const Tensor& self) {
  // TODO(@caizhi): use musa porting to instead putting to cpu.
  c10::musa::MUSAGuard device_guard(self.device());
  if (self.scalar_type() == ScalarType::Double) {
    return at::max(self.to("cpu")).to("musa");
  }
  return MaxAllCall(self, ::musa::dnn::Reduce::Mode::MAX);
}

std::tuple<Tensor, Tensor> MaxDim(
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  return ReductionIndices(self, dim, keepdim, ::musa::dnn::Reduce::Mode::MAX);
}

std::tuple<Tensor&, Tensor&> MaxDimMax(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& output,
    Tensor& indices) {
  dim = maybe_wrap_dim(dim, self.dim());
  IntArrayRef dims(dim);
  DimVector dims_vec(dims);
  maybe_wrap_dims(dims_vec, self.dim());
  auto shape = at::meta::get_reduction_shape(self, dims_vec, keepdim);
  if (0 == output.numel()) {
    at::native::resize_output(output, shape);
  }
  if (0 == indices.numel()) {
    at::native::resize_output(indices, shape);
  }
  ReduceIndicesCall(output, indices, self, dim, ::musa::dnn::Reduce::Mode::MAX);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> MaxNamesDim(
    const Tensor& self,
    Dimname dim,
    bool keepdim) {
  return ReductionIndices(
      self,
      dimname_to_position(self, dim),
      keepdim,
      ::musa::dnn::Reduce::Mode::MAX);
}

std::tuple<Tensor&, Tensor&> MaxNamesDimMax(
    const Tensor& self,
    Dimname dim,
    bool keepdim,
    Tensor& output,
    Tensor& indices) {
  UNUSED(keepdim);
  ReduceIndicesCall(
      output,
      indices,
      self,
      dimname_to_position(self, dim),
      ::musa::dnn::Reduce::Mode::MAX);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

namespace {

struct StructuredMusaAll final : public at::native::structured_all_all_out {
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }

  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
};

at::Tensor WrapperMusaAll(const at::Tensor& self) {
  StructuredMusaAll op;
  op.meta(self);
  op.impl(self, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

struct StructuredMusaAllOut final : public at::native::structured_all_all_out {
  StructuredMusaAllOut(Tensor& out0) : outputs_{std::ref(out0)} {}

  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
    if (C10_UNLIKELY(maybe_proxy.has_value())) {
      proxy_outputs_[output_idx] =
          c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());
    }
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }

  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
};

at::Tensor& WrapperMusaAllOut(const at::Tensor& self, at::Tensor& out) {
  StructuredMusaAllOut op(out);
  op.meta(self);
  op.impl(self, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value()) {
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  }
  return out;
}

} // anonymous namespace

Tensor All(const Tensor& self) {
  if (self.scalar_type() != ScalarType::Bool &&
      self.scalar_type() != ScalarType::Byte) {
    return WrapperMusaAll(self);
  }

  // mtdnn now only support bool, so we need to cast when input_dype=Byte
  if (self.scalar_type() == ScalarType::Byte) {
    Tensor self_ = self.to(ScalarType::Bool);
    auto out = Reduction(
        self_,
        IntArrayRef{},
        false,
        ScalarType::Bool,
        ::musa::dnn::Reduce::Mode::AND);
    out = out.to(ScalarType::Byte);
    return out;
  } else {
    return Reduction(
        self,
        IntArrayRef{},
        false,
        self.scalar_type(),
        ::musa::dnn::Reduce::Mode::AND);
  }
}

Tensor AllDim(const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(
      self.scalar_type() == ScalarType::Bool ||
          self.scalar_type() == ScalarType::Byte || self.item<int>() == 0 ||
          self.item<int>() == 1,
      concatenate(
          "Now only support bool/uint8 type or 0/1 value, but got ",
          self.scalar_type(),
          self));
  IntArrayRef dims(dim);
  if (self.scalar_type() == ScalarType::Byte) {
    Tensor self_;
    self_ = self.to(ScalarType::Bool);
    return Reduction(
        self_,
        {dim},
        keepdim,
        self.scalar_type(),
        ::musa::dnn::Reduce::Mode::AND);
  } else {
    return Reduction(
        self,
        {dim},
        keepdim,
        self.scalar_type(),
        ::musa::dnn::Reduce::Mode::AND);
  }
}

Tensor& AllDimOut(const Tensor& self, int64_t dim, bool keepdim, Tensor& out) {
  UNUSED(keepdim);
  TORCH_CHECK(
      self.scalar_type() == ScalarType::Bool ||
          self.scalar_type() == ScalarType::Byte || self.item<int>() == 0 ||
          self.item<int>() == 1,
      concatenate(
          "Now only support bool/uint8 type or 0/1 value, but got ",
          self.scalar_type(),
          self));
  IntArrayRef dims(dim);
  if (self.scalar_type() == ScalarType::Byte) {
    Tensor self_;
    self_ = self.to(ScalarType::Bool);
    ReduceCall(out, self_, dims, ::musa::dnn::Reduce::Mode::AND);
  } else {
    ReduceCall(out, self, dims, ::musa::dnn::Reduce::Mode::AND);
  }

  return out;
}
void ArgMinOrMaxOutTemplate(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    Tensor& result,
    ::musa::dnn::Reduce::Mode m) {
  Tensor contiguous_self = self.contiguous();
  if (!dim.has_value()) {
    contiguous_self = contiguous_self.flatten();
  }
  Tensor out_data =
      at::empty(result.sizes(), self.options().dtype(self.scalar_type()));
  auto dim_ = dim.has_value() ? maybe_wrap_dim(dim.value(), self.dim()) : 0;
  ReduceIndicesCall(out_data, result, contiguous_self, dim_, m);
}
Tensor& ArgmaxOut(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    Tensor& result) {
  ArgMinOrMaxOutTemplate(
      self, dim, keepdim, result, ::musa::dnn::Reduce::Mode::MAX);
  return result;
}

Tensor& ArgminOut(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    Tensor& result) {
  ArgMinOrMaxOutTemplate(
      self, dim, keepdim, result, ::musa::dnn::Reduce::Mode::MIN);
  return result;
}

Tensor MinAllCall(const Tensor& self, ::musa::dnn::Reduce::Mode m) {
  auto out_dtype = self.scalar_type();
  // torch.min call reudce_all according to out.dim
  Tensor output = at::empty({}, self.options().dtype(out_dtype));
  DimVector dims_vec(0);
  if (self.numel() == 0) {
    output.zero_();
  } else {
    ReduceCall(output, self, dims_vec, m);
  }
  return output;
}

Tensor MinAll(const Tensor& self) {
  // TODO(@caizhi): use musa porting to instead putting to cpu.
  c10::musa::MUSAGuard device_guard(self.device());
  if (self.scalar_type() == ScalarType::Double) {
    return at::min(self.to("cpu")).to("musa");
  }
  return MinAllCall(self, ::musa::dnn::Reduce::Mode::MIN);
}

std::tuple<Tensor, Tensor> MinDim(
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  return ReductionIndices(self, dim, keepdim, ::musa::dnn::Reduce::Mode::MIN);
}

std::tuple<Tensor&, Tensor&> MinDimMin(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& output,
    Tensor& indices) {
  UNUSED(keepdim);
  ReduceIndicesCall(output, indices, self, dim, ::musa::dnn::Reduce::Mode::MIN);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> MinNamesDim(
    const Tensor& self,
    Dimname dim,
    bool keepdim) {
  return ReductionIndices(
      self,
      dimname_to_position(self, dim),
      keepdim,
      ::musa::dnn::Reduce::Mode::MIN);
}

std::tuple<Tensor&, Tensor&> MinNamesDimMin(
    const Tensor& self,
    Dimname dim,
    bool keepdim,
    Tensor& output,
    Tensor& indices) {
  ReduceIndicesCall(
      output,
      indices,
      self,
      dimname_to_position(self, dim),
      ::musa::dnn::Reduce::Mode::MIN);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<at::Tensor, at::Tensor> VarMeanCorrection(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  // No device check
  torch::utils::musa_lazy_init();
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::var_mean(self, dim, correction, keepdim);
}

Tensor& LogSumExpOut(
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    Tensor& result) {
  c10::musa::MUSAGuard device_guard(self.device());
  TORCH_CHECK(
      at::isFloatingType(result.scalar_type()),
      "logsumexp(): Expected floating point type for result tensor, but got: ",
      result.scalar_type());
  namedinference::propagate_names_for_reduction(result, self, dims, keepdim);

  if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    // for integral inputs, promote input to default floating type.
    auto default_dtype = at::typeMetaToScalarType(c10::get_default_dtype());
    at::native::LogSumExpOutImpl(result, self.to(default_dtype), dims);
  } else {
    at::native::LogSumExpOutImpl(result, self, dims);
  }

  return result;
}

Tensor LogSumExp(const Tensor& self, IntArrayRef dims, bool keepdim) {
  TensorOptions result_options;
  DimVector dims_vec(dims);
  maybe_wrap_dims(dims_vec, self.dim());
  auto shape = at::meta::get_reduction_shape(self, dims_vec, keepdim);

  if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    // even for integral inputs, result is floating dtype
    auto default_dtype = at::typeMetaToScalarType(c10::get_default_dtype());
    result_options = self.options().dtype(default_dtype);
  } else {
    result_options = self.options();
  }
  auto result = at::empty(shape, result_options);
  return LogSumExpOut(self, dims_vec, keepdim, result);
}

at::Tensor VarCorrection(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::var(self, dim, correction, keepdim);
}

at::Tensor& VarOutCorrection(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction,
    bool keepdim,
    at::Tensor& out) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::var_out(self, dim, correction, keepdim, out);
}

ADVANCED_REGISTER(aten, PrivateUse1, "mean", Mean)
ADVANCED_REGISTER(aten, PrivateUse1, "mean.dim", MeanDim)
ADVANCED_REGISTER(aten, PrivateUse1, "mean.out", MeanOut)
ADVANCED_REGISTER(aten, PrivateUse1, "mean.names_dim", MeanNamesDim)
ADVANCED_REGISTER(aten, PrivateUse1, "mean.names_out", MeanNamesDimOut)
ADVANCED_REGISTER(aten, PrivateUse1, "sum", Sum)
ADVANCED_REGISTER(aten, PrivateUse1, "sum.IntList_out", SumIntListOut)
ADVANCED_REGISTER(aten, PrivateUse1, "sum.dim_DimnameList", SumDimnameList)
ADVANCED_REGISTER(aten, PrivateUse1, "sum.DimnameList_out", SumDimnameListOut)
ADVANCED_REGISTER(aten, PrivateUse1, "sum.dim_IntList", SumIntList)

ADVANCED_REGISTER(aten, PrivateUse1, "prod", Prod)
ADVANCED_REGISTER(aten, PrivateUse1, "prod.int_out", ProdIntOut)

ADVANCED_REGISTER(aten, PrivateUse1, "norm.out", NormOut)
ADVANCED_REGISTER(aten, PrivateUse1, "norm.dtype_out", NormDtypeOut)

ADVANCED_REGISTER(aten, PrivateUse1, "cumsum", Cumsum)
ADVANCED_REGISTER(aten, PrivateUse1, "cumsum_", Cumsum_)
ADVANCED_REGISTER(aten, PrivateUse1, "cumsum.out", Cumsum_Out)

ADVANCED_REGISTER(aten, PrivateUse1, "any", Any)
ADVANCED_REGISTER(aten, PrivateUse1, "any.all_out", AnyOut)
ADVANCED_REGISTER(aten, PrivateUse1, "any.dim", AnyDim)
ADVANCED_REGISTER(aten, PrivateUse1, "any.out", AnyDimOut)

ADVANCED_REGISTER(aten, PrivateUse1, "max", MaxAll)
ADVANCED_REGISTER(aten, PrivateUse1, "max.dim", MaxDim)
ADVANCED_REGISTER(aten, PrivateUse1, "max.dim_max", MaxDimMax)
ADVANCED_REGISTER(aten, PrivateUse1, "max.names_dim", MaxNamesDim)
ADVANCED_REGISTER(aten, PrivateUse1, "max.names_dim_max", MaxNamesDimMax)

ADVANCED_REGISTER(aten, PrivateUse1, "min", MinAll)
ADVANCED_REGISTER(aten, PrivateUse1, "min.dim", MinDim)
ADVANCED_REGISTER(aten, PrivateUse1, "min.dim_min", MinDimMin)
ADVANCED_REGISTER(aten, PrivateUse1, "min.names_dim", MinNamesDim)
ADVANCED_REGISTER(aten, PrivateUse1, "min.names_dim_min", MinNamesDimMin)

ADVANCED_REGISTER(aten, PrivateUse1, "all", All)
ADVANCED_REGISTER(aten, PrivateUse1, "all.dim", AllDim)
ADVANCED_REGISTER(aten, PrivateUse1, "all.out", AllDimOut)
ADVANCED_REGISTER(aten, PrivateUse1, "argmax.out", ArgmaxOut)
ADVANCED_REGISTER(aten, PrivateUse1, "argmin.out", ArgminOut)

ADVANCED_REGISTER(aten, PrivateUse1, "var_mean.correction", VarMeanCorrection)
ADVANCED_REGISTER(aten, PrivateUse1, "var.correction", VarCorrection)
ADVANCED_REGISTER(aten, PrivateUse1, "var.correction_out", VarOutCorrection)

ADVANCED_REGISTER(aten, PrivateUse1, "logsumexp", LogSumExp)
ADVANCED_REGISTER(aten, PrivateUse1, "logsumexp.out", LogSumExpOut)

} // namespace musa
} // namespace at
