#include <ATen/Config.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/scatter.h>
#endif

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace musa {

Tensor& musaScatterOut(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  Tensor index_cpu = index.cpu();
  Tensor self_cpu = self.cpu();
  Tensor src_cpu = src.cpu();
  out = self_cpu.scatter_(dim, index_cpu, src_cpu).to(self.device());

  // TODO(kang.chen): when muDNN support long, we will use it.
  // muHandle h;
  // musa::dnn::Scatter op;
  // out = self;
  // CHECK_MUDNN_STATUS(op.SetMode(musa::dnn::Scatter::Mode::UPDATE_ONLY),
  // "SetMode"); auto in_mt = CreateMUTensor(self); auto idx_mt =
  // CreateMUTensor(index); auto src_mt = CreateMUTensor(src); auto out_mt =
  // CreateMUTensor(out); CHECK_MUDNN_STATUS(op.Run(h, out_mt, idx_mt, src_mt,
  // dim, InternalMemAlloc), "Run");
  return out;
}

Tensor musaScatter(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  auto self_ = self.cpu();
  auto contiguous_index = index.cpu();
  auto src_ = src.cpu();

  auto out_ = at::scatter(self_, dim, contiguous_index, src_);
  auto out = out_.to(self.device());
  return out;
}

Tensor& ScatterAddOut(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of scatter_add must be MUSA, but now it is ",
      self.device());
  TORCH_CHECK(
      index.device().type() == kMUSA,
      "Device of index tensor of scatter_add must be MUSA, but now it is ",
      index.device());
  TORCH_CHECK(
      src.device().type() == kMUSA,
      "Device of src tensor of scatter_add must be MUSA, but now it is ",
      src.device());
  TORCH_CHECK(
      out.device().type() == kMUSA,
      "Device of out tensor of scatter_add must be MUSA, but now it is ",
      out.device());

  TORCH_CHECK(
      self.scalar_type() == out.scalar_type(),
      "Dtype of input tensor of scatter_add should be same as out, which is ",
      self.scalar_type(),
      ", and out dtype is ",
      out.scalar_type());
  TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long ||
          index.scalar_type() == at::ScalarType::Int,
      "Dtype of index tensor of scatter_add only support Long/Int, but "
      "now it is ",
      index.scalar_type());
  TORCH_CHECK(
      src.scalar_type() == at::ScalarType::Float ||
          src.scalar_type() == at::ScalarType::Long,
      "Dtype of src tensor of scatter_add only support Float32/Long, but now it is ",
      src.scalar_type());
  TORCH_CHECK(
      out.scalar_type() == at::ScalarType::Float ||
          out.scalar_type() == at::ScalarType::Long,
      "Dtype of out tensor of scatter_add only support Float32/Long, but now it is ",
      out.scalar_type());
  c10::musa::MUSAGuard device_guard(self.device());
  if (dim < 0) {
    dim += self.dim();
  }
  Tensor self_ = self.contiguous();
  Tensor src_ = src.contiguous();
  Tensor contiguous_index = index.contiguous();
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Scatter op;
  CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Scatter::Mode::ADD), "SetMode");
  auto self_mt = CreateMUTensor(self_);
  auto out_mt = CreateMUTensor(out);
  auto idx_mt = CreateMUTensor(contiguous_index);
  auto src_mt = CreateMUTensor(src_);
  CHECK_MUDNN_STATUS(
      op.Run(h, out_mt, self_mt, idx_mt, src_mt, dim, InternalMemAlloc), "Run");
  return out;
}

Tensor& ScatterAddU(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of scatter_add must be MUSA, but now it is ",
      self.device());
  TORCH_CHECK(
      index.device().type() == kMUSA,
      "Device of index tensor of scatter_add must be MUSA, but now it is ",
      index.device());
  TORCH_CHECK(
      src.device().type() == kMUSA,
      "Device of src tensor of scatter_add must be MUSA, but now it is ",
      src.device());

  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of scatter_add only support Float32, but "
      "now it is ",
      self.scalar_type());
  TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long ||
          index.scalar_type() == at::ScalarType::Int,
      "Dtype of index tensor of scatter_add only support Long/Int, but "
      "now it is ",
      index.scalar_type());
  TORCH_CHECK(
      src.scalar_type() == at::ScalarType::Float,
      "Dtype of src tensor of scatter_add only support Float32, but now it is ",
      src.scalar_type());
  c10::musa::MUSAGuard device_guard(self.device());
  if (dim < 0) {
    dim += self.dim();
  }
  Tensor self_ = self.contiguous();
  Tensor src_ = src.contiguous();
  Tensor contiguous_index = index.contiguous();
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Scatter op;
  CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Scatter::Mode::ADD), "SetMode");
  auto self_mt = CreateMUTensor(self);
  auto idx_mt = CreateMUTensor(index);
  auto src_mt = CreateMUTensor(src);
  CHECK_MUDNN_STATUS(
      op.Run(h, self_mt, idx_mt, src_mt, dim, InternalMemAlloc), "Run");
  return self;
}

at::Tensor& ScatterValueOut(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value,
    at::Tensor& out) {
  const Tensor& self_cpu = self.cpu();
  const Tensor& index_cpu = index.cpu();
  auto cpu_result = ::at::scatter(self_cpu, dim, index_cpu, value);
  out.copy_(cpu_result);
  return out;
}

ADVANCED_REGISTER(aten, PrivateUse1, "scatter.src_out", musaScatterOut)
ADVANCED_REGISTER(aten, PrivateUse1, "scatter.src", musaScatter)
ADVANCED_REGISTER(aten, PrivateUse1, "scatter.value_out", ScatterValueOut)
ADVANCED_REGISTER(aten, PrivateUse1, "scatter_add.out", ScatterAddOut)
ADVANCED_REGISTER(aten, PrivateUse1, "scatter_add_", ScatterAddU)

} // namespace musa
} // namespace at
