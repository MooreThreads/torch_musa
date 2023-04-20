#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace native {
namespace musa {

void TriCallOut(
    Tensor& out_,
    const Tensor& in_,
    const bool upper,
    const int64_t diag) {
  muHandle h;
  auto out = CreateMUTensor(out_);
  auto in = CreateMUTensor(in_);
  ::musa::dnn::TriangularMat op;

  op.SetMode(
      upper ? ::musa::dnn::TriangularMat::Mode::TRIU
            : ::musa::dnn::TriangularMat::Mode::TRIL);
  op.SetDiagonal(diag);

  CHECK_MUDNN_STATUS(op.Run(h, out, in), "Run");
}

Tensor TriCall(Tensor& self, const bool upper, const int64_t diag) {
  Tensor self_ = Contiguous(self);
  Tensor output = empty_mtgpu(
      self.sizes().vec(),
      self.scalar_type(),
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  TriCallOut(output, self, upper, diag);
  return self.copy_(output);
}

template <bool upper>
Tensor Tri(const Tensor& self, int64_t diagonal = 0) {
  Tensor self_ = Contiguous(self);
  Tensor output = empty_mtgpu(
      self.sizes().vec(),
      self.scalar_type(),
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  TriCallOut(output, self_, upper, diagonal);
  return output;
}

template <bool upper>
Tensor& Tri_(Tensor& self, int64_t diagonal = 0) {
  TriCall(self, upper, diagonal);
  return self;
}

template <bool upper>
Tensor& TriOut(const Tensor& self, int64_t diagonal, Tensor& output) {
  Tensor self_ = Contiguous(self);
  TriCallOut(output, self_, upper, diagonal);
  return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("triu", &Tri<true>);
  m.impl("triu_", &Tri_<true>);
  m.impl("triu.out", &TriOut<true>);
  m.impl("tril", &Tri<false>);
  m.impl("tril_", &Tri_<false>);
  m.impl("tril.out", &TriOut<false>);
}

} // namespace musa
} // namespace native
} // namespace at
