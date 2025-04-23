// Functions that fill Tensors with constants.
#include <ATen/Config.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Fill.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/ops/masked_fill_native.h>
#include <ATen/ops/ones.h>
#include <ATen/quantized/QTensorImpl.h>
#endif

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

Tensor& QTensorCopy(Tensor& self, const Tensor& src) {
  TORCH_CHECK(
      src.scalar_type() == at::kFloat,
      "Quantized copy only works with kFloat as source Tensor");
  TORCH_CHECK(
      (self.is_contiguous() && src.is_contiguous()) ||
          (self.is_contiguous(at::MemoryFormat::ChannelsLast) &&
           src.is_contiguous(at::MemoryFormat::ChannelsLast)),
      "Quantized copy only works with contiguous and NHWC Tensors");
  TORCH_CHECK(
      self.sizes().equals(src.sizes()),
      "Quantized copy only works with Tensor with the same shape");
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "Copy", [&]() {
    if (self.qscheme() == kPerChannelAffine ||
        self.qscheme() == kPerChannelAffineFloatQParams ||
        self.qscheme() == kPerChannelSymmetric) {
      at::native::quantize_tensor_per_channel_affine(
          src,
          self,
          self.q_per_channel_scales(),
          self.q_per_channel_zero_points(),
          self.q_per_channel_axis());
    } else {
      at::native::quantize_tensor_per_tensor_affine(
          src, self, self.q_scale(), self.q_zero_point());
    }
  });
  return self;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
static Tensor& FillOutQuantized(Tensor& self, const Scalar& value) {
  at::Tensor out = at::ones(self.sizes()).to(kFloat) * value;
  out = out.to(self.device()).to(self.suggest_memory_format());
  // TODO(@fan.mo): Fix copy_ error of musa quantized tensor
  self = QTensorCopy(self, out);
  return self;
}

Tensor& FillQuantizedScalar(Tensor& self, const Scalar& value) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return FillOutQuantized(self, value);
}

Tensor& FillQuantizedTensor(Tensor& self, const Tensor& value) {
  const OptionalDeviceGuard device_guard(device_of(self));
  TORCH_CHECK(
      value.dim() == 0,
      "fill_ only supports 0-dimension value tensor but got tensor with ",
      value.dim(),
      " dimensions.");
  return FillOutQuantized(self, value.item());
}

} // namespace musa
} // namespace at
