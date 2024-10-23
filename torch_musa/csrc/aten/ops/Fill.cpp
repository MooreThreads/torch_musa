#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

Tensor& FillOp(
    Tensor& self,
    const Scalar& value,
    const c10::optional<Tensor>& mask = c10::nullopt) {
  if C10_UNLIKELY (self.numel() == 0) {
    return self;
  }
  c10::musa::MUSAGuard device_guard(self.device());
  at::musa::muHandle& h = GetMudnnHandle();
  auto self_mu = at::musa::CreateMUTensor(self);

  ::musa::dnn::Fill op;
  const auto fill_type = self.scalar_type();
  if (fill_type == ScalarType::Bool) {
    const auto fill_value = static_cast<int64_t>(value.to<bool>());
    CHECK_MUDNN_STATUS(op.SetValue(fill_value), "SetValue");
  } else if (isIntegralType(fill_type, false)) {
    CHECK_MUDNN_STATUS(op.SetValue(value.toLong()), "SetValue");
  } else {
    CHECK_MUDNN_STATUS(op.SetValue(value.toDouble()), "SetValue");
  }
  if (mask.has_value()) {
    auto mask_mu = at::musa::CreateMUTensor(mask.value());
    CHECK_MUDNN_STATUS(op.Run(h, self_mu, mask_mu), "RunMask");
  } else {
    CHECK_MUDNN_STATUS(op.Run(h, self_mu), "Run");
  }
  return self;
}

Tensor& Fill(Tensor& self, const Scalar& value) {
  return FillOp(self, value);
}

Tensor& Fill_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(
      value.dim() == 0,
      "fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ",
      value.dim(),
      " dimension(s).");
  return FillOp(self, value.item());
}

Tensor& Zero_(Tensor& self) {
  return FillOp(self, 0.0);
}

Tensor& MaskedFill(Tensor& self, const Tensor& mask, const Scalar& value) {
  return FillOp(self, value, mask);
}

Tensor& MaskedFillTensor(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  TORCH_CHECK(
      value.dim() == 0,
      "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ",
      value.dim(),
      " dimension(s).");
  // We hit this function if either of the input tensor lives on MUSA.
  // It is ok, if `value` is `CPU` tensor but we should not allow `self` or
  // `mask` to be CPU tensor. Check for `self` and `mask` being on same device
  // exists in `masked_fill__cuda` (Scalar version).
  TORCH_CHECK(
      !self.device().is_cpu(),
      "masked_fill_: Expected inputs to be on same device")
  return FillOp(self, value.item(), mask);
}

} // namespace musa
} // namespace at
