#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace native {
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
  auto input = Contiguous(self);
  auto out = CreateMUTensor(output);
  auto in = CreateMUTensor(input);

  muHandle h;
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
  out_dtype = musa_infer_dtype_from_optional(self, out_dtype, Tensor());
  DimVector dims_(dim);
  maybe_wrap_dims(dims_, self.dim());
  auto shape = at::meta::get_reduction_shape(self, dims_, keepdim);

  Tensor output = at::empty(shape, self.options().dtype(out_dtype));
  namedinference::propagate_names_for_reduction(output, self, dims_, keepdim);

  if (self.numel() == 0) {
    output.zero_();
  } else {
    ReduceCall(output, self, dims_, m, p, is_norm);
  }
  return output;
}

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

Tensor& NormDtypeOut(
    const Tensor& self,
    const c10::optional<at::Scalar>& p,
    at::IntArrayRef dim,
    bool keepdim,
    at::ScalarType dtype,
    at::Tensor& out) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of Norm.out must be MUSA, but now is ",
      self.device());
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of Norm.out only support Float32, ",
      "but now it is ",
      self.scalar_type());
  TORCH_CHECK(
      dtype == at::ScalarType::Float,
      "Dtype of input tensor of Norm.out only support Float32, ",
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
  muTensor self_mt = CreateMUTensor(self);
  muTensor out_mt = CreateMUTensor(out);

  ::musa::dnn::Handle h;
  ::musa::dnn::Cumsum csop;
  CHECK_MUDNN_STATUS(csop.SetDim(dim), "SetDim");
  CHECK_MUDNN_STATUS(csop.Run(h, out_mt, self_mt, InternalMemAlloc), "Run");
  return out;
}

Tensor Cumsum(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype_opt) {
  Tensor self_ = Contiguous(self);
  auto out = at::empty_like(self);
  return CumsumCall(self_, dim, dtype_opt, out);
}

Tensor& Cumsum_(
    Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype_opt) {
  Tensor self_ = Contiguous(self);
  auto out = self;
  CumsumCall(self_, dim, dtype_opt, out);
  return self;
}

Tensor& Cumsum_Out(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype_opt,
    Tensor& out) {
  Tensor self_ = Contiguous(self);
  CumsumCall(self_, dim, dtype_opt, out);
  return out;
}

Tensor Any(const Tensor& self) {
  TORCH_CHECK(
      self.scalar_type() == ScalarType::Bool, "Now only support bool type");
  return Reduction(
      self,
      IntArrayRef{},
      false,
      self.scalar_type(),
      ::musa::dnn::Reduce::Mode::OR);
}

Tensor& AnyOut(const Tensor& self, Tensor& out) {
  TORCH_CHECK(
      self.scalar_type() == ScalarType::Bool, "Now only support bool type");
  IntArrayRef dims = {};
  ReduceCall(out, self, dims, ::musa::dnn::Reduce::Mode::OR);
  return out;
}

Tensor AnyDim(const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(
      self.scalar_type() == ScalarType::Bool, "Now only support bool type");
  IntArrayRef dims(dim);
  return Reduction(
      self, {dim}, keepdim, self.scalar_type(), ::musa::dnn::Reduce::Mode::OR);
}

Tensor& AnyDimOut(const Tensor& self, int64_t dim, bool keepdim, Tensor& out) {
  UNUSED(keepdim);
  TORCH_CHECK(
      self.scalar_type() == ScalarType::Bool, "Now only support bool type");
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

  auto input = Contiguous(self);

  auto out = CreateMUTensor(output);
  auto ids = CreateMUTensor(indices);
  auto in = CreateMUTensor(input);

  muHandle h;
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
  DimVector dims_(dims);
  maybe_wrap_dims(dims_, self.dim());
  auto shape = at::meta::get_reduction_shape(self, dims_, keepdim);

  auto out_dtype = self.scalar_type();
  Tensor output = at::empty(shape, self.options().dtype(out_dtype));
  Tensor indices = at::empty(shape, self.options().dtype(kLong));
  namedinference::propagate_names_for_reduction(output, self, dims_, keepdim);
  namedinference::propagate_names_for_reduction(indices, self, dims_, keepdim);

  ReduceIndicesCall(output, indices, self, dim, m);
  return std::make_tuple(output, indices);
}

Tensor MaxAllCall(const Tensor& self, ::musa::dnn::Reduce::Mode m) {
  auto out_dtype = self.scalar_type();
  // torch.max call reudce_all according to out.dim
  Tensor output = at::empty({}, self.options().dtype(out_dtype));
  DimVector dims_(0);
  if (self.numel() == 0) {
    output.zero_();
  } else {
    ReduceCall(output, self, dims_, m);
  }
  return output;
}

Tensor MaxAll(const Tensor& self) {
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
  UNUSED(keepdim);
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

Tensor All(const Tensor& self) {
  TORCH_CHECK(
      self.scalar_type() == ScalarType::Bool ||
          self.scalar_type() == ScalarType::Byte,
      "Now only support bool/uint8 type");
  // mtdnn now only support bool, so we need to cast when input_dype=Byte
  if (self.scalar_type() == ScalarType::Byte) {
    Tensor self_;
    self_ = self.to(ScalarType::Bool);
    return Reduction(
        self_,
        IntArrayRef{},
        false,
        self.scalar_type(),
        ::musa::dnn::Reduce::Mode::AND);
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
          self.scalar_type() == ScalarType::Byte,
      "Now only support bool/uint8 type");
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
          self.scalar_type() == ScalarType::Byte,
      "Now only support bool/uint8 type");
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
// TODO(zaixing.wang): mudnn ReduceIndices only support float, int64 input
Tensor& ArgmaxOut(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    Tensor& result) {
  Tensor contiguous_self = Contiguous(self);
  if (!dim.has_value()) {
    contiguous_self = contiguous_self.flatten();
  }
  Tensor out_data =
      at::empty(result.sizes(), self.options().dtype(self.scalar_type()));
  auto dim_ = dim.has_value() ? maybe_wrap_dim(dim.value(), self.dim()) : 0;
  ReduceIndicesCall(
      out_data, result, contiguous_self, dim_, ::musa::dnn::Reduce::Mode::MAX);
  return result;
}
#pragma GCC diagnostic pop

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mean", &Mean);
  m.impl("mean.dim", &MeanDim);
  m.impl("mean.out", &MeanOut);
  m.impl("mean.names_dim", &MeanNamesDim);
  m.impl("mean.names_out", &MeanNamesDimOut);
  m.impl("sum", &Sum);
  m.impl("sum.IntList_out", &SumIntListOut);
  m.impl("sum.dim_DimnameList", &SumDimnameList);
  m.impl("sum.DimnameList_out", &SumDimnameListOut);
  m.impl("sum.dim_IntList", &SumIntList);

  m.impl("norm.out", &NormOut);
  m.impl("norm.dtype_out", &NormDtypeOut);

  m.impl("cumsum", &Cumsum);
  m.impl("cumsum_", &Cumsum_);
  m.impl("cumsum.out", &Cumsum_Out);

  m.impl("any", &Any);
  m.impl("any.all_out", &AnyOut);
  m.impl("any.dim", &AnyDim);
  m.impl("any.out", &AnyDimOut);

  m.impl("max", &MaxAll);
  m.impl("max.dim", &MaxDim);
  m.impl("max.dim_max", &MaxDimMax);
  m.impl("max.names_dim", &MaxNamesDim);
  m.impl("max.names_dim_max", &MaxNamesDimMax);

  m.impl("all", &All);
  m.impl("all.dim", &AllDim);
  m.impl("all.out", &AllDimOut);
  m.impl("argmax.out", &ArgmaxOut);
}

} // namespace musa
} // namespace native
} // namespace at
