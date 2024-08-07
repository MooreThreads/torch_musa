#include "torch_musa/csrc/aten/ops/Triangular.h"
#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/tril_native.h>
#include <ATen/ops/triu_native.h>
#endif
#include <string>

using at::native::TriangularMode;

namespace at {
namespace native {

DEFINE_DISPATCH(triu_stub);
REGISTER_NO_CPU_DISPATCH(triu_stub);
DEFINE_DISPATCH(tril_stub);
REGISTER_NO_CPU_DISPATCH(tril_stub);

} // namespace native

namespace musa {

void TriCallOut(
    Tensor& out,
    const Tensor& input,
    TriangularMode mode,
    const int64_t diag,
    const std::string name) {
  // TODO(@mt-ai/mt-sw-compute): this kernel now doesn't support dim>=8
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of " + name + " must be MUSA");
  TORCH_CHECK(
      input.dim() < 8, "For dim>=8, float64 and int64 dtype would fail to run");

  // NOTE: since this kernel doesn't support non-contiguous tensors,
  // we have to make sure output and input are both contiguous tensors
  if (mode == TriangularMode::TRIU) {
    at::native::triu_stub(kMUSA, out, input, diag);
  } else {
    at::native::tril_stub(kMUSA, out, input, diag);
  }
}

Tensor Triu(const Tensor& self, int64_t diagonal) {
  c10::musa::MUSAGuard device_guard(self.device());

  Tensor output = at::empty_like(
      self, self.options().memory_format(c10::MemoryFormat::Contiguous));
  if (!self.is_contiguous()) {
    Tensor self_contig = self.contiguous();
    TriCallOut(output, self_contig, TriangularMode::TRIU, diagonal, "Triu");
  } else {
    TriCallOut(output, self, TriangularMode::TRIU, diagonal, "Triu");
  }
  return output;
}

Tensor& Triu_(Tensor& self, int64_t diagonal) {
  c10::musa::MUSAGuard device_guard(self.device());

  if (!self.is_contiguous()) {
    Tensor self_contig = self.contiguous();
    TriCallOut(
        self_contig, self_contig, TriangularMode::TRIU, diagonal, "Triu");
    self.copy_(self_contig);
  } else {
    TriCallOut(self, self, TriangularMode::TRIU, diagonal, "Triu");
  }

  return self;
}

Tensor& TriuOut(const Tensor& self, int64_t diagonal, Tensor& output) {
  c10::musa::MUSAGuard device_guard(self.device());

  output.resize_(self.sizes());
  Tensor out_contig = output.contiguous();

  if (!self.is_contiguous()) {
    Tensor self_contig = self.contiguous();
    TriCallOut(out_contig, self_contig, TriangularMode::TRIU, diagonal, "Triu");
  } else {
    TriCallOut(out_contig, self, TriangularMode::TRIU, diagonal, "Triu");
  }

  output.copy_(out_contig);
  return output;
}

Tensor Tril(const Tensor& self, int64_t diagonal) {
  c10::musa::MUSAGuard device_guard(self.device());

  Tensor output = at::empty_like(
      self, self.options().memory_format(c10::MemoryFormat::Contiguous));
  if (!self.is_contiguous()) {
    Tensor self_contig = self.contiguous();
    TriCallOut(output, self_contig, TriangularMode::TRIL, diagonal, "Tril");
  } else {
    TriCallOut(output, self, TriangularMode::TRIL, diagonal, "Tril");
  }
  return output;
}

Tensor& Tril_(Tensor& self, int64_t diagonal) {
  c10::musa::MUSAGuard device_guard(self.device());

  if (!self.is_contiguous()) {
    Tensor self_contig = self.contiguous();
    TriCallOut(
        self_contig, self_contig, TriangularMode::TRIL, diagonal, "Tril");
    self.copy_(self_contig);
  } else {
    TriCallOut(self, self, TriangularMode::TRIL, diagonal, "Tril");
  }

  return self;
}

Tensor& TrilOut(const Tensor& self, int64_t diagonal, Tensor& output) {
  c10::musa::MUSAGuard device_guard(self.device());

  output.resize_(self.sizes());
  Tensor out_contig = output.contiguous();

  if (!self.is_contiguous()) {
    Tensor self_contig = self.contiguous();
    TriCallOut(out_contig, self_contig, TriangularMode::TRIL, diagonal, "Tril");
  } else {
    TriCallOut(out_contig, self, TriangularMode::TRIL, diagonal, "Tril");
  }

  output.copy_(out_contig);
  return output;
}

} // namespace musa
} // namespace at
