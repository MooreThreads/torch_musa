#include "torch_musa/csrc/aten/ops/Triangular.h"
#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

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
  // TODO(@mt-ai/mt-sw-compute): this kernel now doesn't support bool dtype and
  // tensor with dim>=8
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of " + name + " must be MUSA");
  TORCH_CHECK(
      input.dim() < 8, "For dim>=8, float64 and int64 dtype would fail to run");

  Tensor input_contiguous = input.contiguous();

  if (mode == TriangularMode::TRIU) {
    at::native::triu_stub(kMUSA, out, input_contiguous, diag);
  } else {
    at::native::tril_stub(kMUSA, out, input_contiguous, diag);
  }
}

Tensor Triu(const Tensor& self, int64_t diagonal = 0) {
  c10::musa::MUSAGuard device_guard(self.device());
  Tensor output = at::empty_like(
      self, self.options().memory_format(c10::MemoryFormat::Contiguous));
  TriCallOut(output, self, TriangularMode::TRIU, diagonal, "Triu");
  return output;
}

Tensor& Triu_(Tensor& self, int64_t diagonal = 0) {
  c10::musa::MUSAGuard device_guard(self.device());
  TriCallOut(self, self, TriangularMode::TRIU, diagonal, "Triu");
  return self;
}

Tensor& TriuOut(const Tensor& self, int64_t diagonal, Tensor& output) {
  c10::musa::MUSAGuard device_guard(self.device());
  output.resize_(self.sizes());
  TriCallOut(output, self, TriangularMode::TRIU, diagonal, "Triu");
  return output;
}

Tensor Tril(const Tensor& self, int64_t diagonal = 0) {
  c10::musa::MUSAGuard device_guard(self.device());
  Tensor output = at::empty_like(
      self, self.options().memory_format(c10::MemoryFormat::Contiguous));
  TriCallOut(output, self, TriangularMode::TRIL, diagonal, "Tril");
  return output;
}

Tensor& Tril_(Tensor& self, int64_t diagonal = 0) {
  c10::musa::MUSAGuard device_guard(self.device());
  TriCallOut(self, self, TriangularMode::TRIL, diagonal, "Tril");
  return self;
}

Tensor& TrilOut(const Tensor& self, int64_t diagonal, Tensor& output) {
  c10::musa::MUSAGuard device_guard(self.device());
  output.resize_(self.sizes());
  TriCallOut(output, self, TriangularMode::TRIL, diagonal, "Tril");
  return output;
}

ADVANCED_REGISTER(aten, PrivateUse1, "triu", Triu)
ADVANCED_REGISTER(aten, PrivateUse1, "triu_", Triu_)
ADVANCED_REGISTER(aten, PrivateUse1, "triu.out", TriuOut)

ADVANCED_REGISTER(aten, PrivateUse1, "tril_", Tril_)
ADVANCED_REGISTER(aten, PrivateUse1, "tril.out", TrilOut)
ADVANCED_REGISTER(aten, PrivateUse1, "tril", Tril)

} // namespace musa
} // namespace at
