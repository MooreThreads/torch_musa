#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

void GluCall(Tensor& o, const Tensor& i, int dim) {
  c10::musa::MUSAGuard device_guard(i.device());
  auto in = CreateMUTensor(i);
  auto out = CreateMUTensor(o);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Glu op;
  CHECK_MUDNN_STATUS(op.SetAxis(dim), "SetAxis");
  CHECK_MUDNN_STATUS(op.Run(h, out, in), "Run");
  return;
}

int64_t CheckGluDim(const Tensor& in, int64_t dim) {
  TORCH_CHECK(in.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, in.dim());
  return wrap_dim;
}

Tensor Glu(const Tensor& self, int64_t dim) {
  dim = CheckGluDim(self, dim);
  auto output_size = self.sizes().vec();
  output_size[dim] = output_size[dim] / 2;
  Tensor self_ = Contiguous(self);
  auto out = empty_mtgpu(
      output_size,
      self_.scalar_type(),
      c10::nullopt,
      self.device(),
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  GluCall(out, self_, static_cast<int>(dim));
  return out;
}

Tensor& GluOut(const Tensor& self, int64_t dim, Tensor& out) {
  Tensor self_ = Contiguous(self);
  dim = CheckGluDim(self_, dim);
  GluCall(out, self_, static_cast<int>(dim));
  return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("glu", &Glu);
  m.impl("glu.out", &GluOut);
}

} // namespace musa
} // namespace at
