#include <ATen/Config.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/native/ScatterGatherChecks.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/scatter.h>
#endif

#include <ATen/ops/view_as_real.h>
#include "torch_musa/csrc/aten/mudnn/Scatter.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/MudnnUtils.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>
#include <string>

namespace at {
namespace musa {

using Mode = ::musa::dnn::Scatter::Mode;

namespace {
inline void ScatterMetaCheck(
    const Tensor& self,
    int64_t& dim,
    const Tensor& index,
    const Tensor& src) {
  dim = at::maybe_wrap_dim(dim, self.dim());
  at::native::scatter_gather_dtype_check("scatter", self, index, src);
  at::native::scatter_shape_check(self, dim, index, src);
}
} // namespace

Tensor& ScatterComplexOp(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out,
    Mode mode);

Tensor& ScatterOp(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out,
    Mode mode) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "self tenosr of ScatterOut should be on MUSA");
  TORCH_CHECK(
      index.device().type() == kMUSA,
      "index tenosr of ScatterOut should be on MUSA");
  TORCH_CHECK(
      src.device().type() == kMUSA,
      "src tenosr of ScatterOut should be on MUSA");
  TORCH_CHECK(
      out.device().type() == kMUSA,
      "out tenosr of ScatterOut should be on MUSA");
  if (C10_UNLIKELY(self.numel() == 0)) {
    return out;
  }
  c10::musa::MUSAGuard device_guard(self.device());
  ScatterMetaCheck(self, dim, index, src);

  if (!out.is_same(self)) {
    out.copy_(self);
  }
  if (index.numel() == 0) {
    return out;
  }

  if (self.is_complex()) {
    return ScatterComplexOp(self, dim, index, src, out, mode);
  }

  Tensor src_ = src.contiguous();
  Tensor index_ = index.contiguous();

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Scatter op;
  ::at::musa::SetScatter(op, mode);

  // mudnn scatter op only supports contig tensors
  // and we always run inplace kernel to avoid `self` tensor contiguous copy
  if (!out.is_contiguous()) {
    Tensor out_ = out.contiguous();
    auto idx_mt = CreateMUTensor(index_);
    auto src_mt = CreateMUTensor(src_);
    auto out_mt = CreateMUTensor(out_);
    CHECK_MUDNN_STATUS(
        op.Run(h, out_mt, idx_mt, src_mt, dim, InternalMemAlloc), "Run");
    out.copy_(out_);
  } else {
    auto idx_mt = CreateMUTensor(index_);
    auto src_mt = CreateMUTensor(src_);
    auto out_mt = CreateMUTensor(out);
    CHECK_MUDNN_STATUS(
        op.Run(h, out_mt, idx_mt, src_mt, dim, InternalMemAlloc), "Run");
  }

  return out;
}

Tensor& ScatterComplexOp(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out,
    Mode mode) {
  auto self_real = at::view_as_real(self);
  auto src_real = at::view_as_real(src);
  auto out_real = at::view_as_real(out);

  for (int64_t component = 0; component < 2; ++component) {
    Tensor self_component = self_real.select(-1, component);
    Tensor src_component = src_real.select(-1, component);
    Tensor out_component = out_real.select(-1, component);

    at::musa::ScatterOp(
        self_component, dim, index, src_component, out_component, mode);
  }
  return out;
}

Tensor& ScatterOut(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long ||
          index.scalar_type() == at::ScalarType::Int,
      "Dtype of index tensor of scatter.out only support Long/Int, but "
      "now it is ",
      index.scalar_type());
  return ScatterOp(self, dim, index, src, out, Mode::UPDATE_ONLY);
}

Tensor& Scatter_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long ||
          index.scalar_type() == at::ScalarType::Int,
      "Dtype of index tensor of scatter_ only support Long/Int, but "
      "now it is ",
      index.scalar_type());
  return ScatterOp(self, dim, index, src, self, Mode::UPDATE_ONLY);
}

Tensor Scatter(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long ||
          index.scalar_type() == at::ScalarType::Int,
      "Dtype of index tensor of scatter only support Long/Int, but "
      "now it is ",
      index.scalar_type());
  Tensor out = at::empty_like(self);
  return ScatterOp(self, dim, index, src, out, Mode::UPDATE_ONLY);
}

at::Tensor& ScatterValueOut(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value,
    at::Tensor& out) {
  TORCH_CHECK(
      self.scalar_type() == out.scalar_type(),
      "Dtype of input tensor of scatter_add should be same as out, which is ",
      self.scalar_type(),
      ", and out dtype is ",
      out.scalar_type());
  TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long ||
          index.scalar_type() == at::ScalarType::Int,
      "Dtype of index tensor of scatter.out only support Long/Int, but "
      "now it is ",
      index.scalar_type());
  Tensor src = at::empty_like(index, self.scalar_type());
  src.fill_(value);
  return ScatterOp(self, dim, index, src, out, Mode::UPDATE_ONLY);
}

at::Tensor ScatterValue(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value) {
  Tensor out = at::empty_like(self);
  return ScatterValueOut(self, dim, index, value, out);
}

at::Tensor& ScatterValue_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value) {
  return ScatterValueOut(self, dim, index, value, self);
}

Tensor& ScatterAddOut(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  TORCH_CHECK(
      self.scalar_type() == src.scalar_type() &&
          self.scalar_type() == out.scalar_type(),
      "self, src and out tensor should be the same");
  TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long ||
          index.scalar_type() == at::ScalarType::Int,
      "Dtype of index tensor of scatter_add.out only support Long/Int, but "
      "now it is ",
      index.scalar_type());
  return ScatterOp(self, dim, index, src, out, Mode::ADD);
}

Tensor& ScatterAdd_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long ||
          index.scalar_type() == at::ScalarType::Int,
      "Dtype of index tensor of scatter_add_ only support Long/Int, but "
      "now it is ",
      index.scalar_type());
  return ScatterOp(self, dim, index, src, self, Mode::ADD);
}

Tensor ScatterAdd(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long ||
          index.scalar_type() == at::ScalarType::Int,
      "Dtype of index tensor of scatter_add only support Long/Int, but "
      "now it is ",
      index.scalar_type());

  Tensor out = at::empty_like(self);
  out = ScatterOp(self, dim, index, src, out, Mode::ADD);
  return out;
}

} // namespace musa
} // namespace at
