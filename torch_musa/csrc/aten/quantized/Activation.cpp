#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/quantized/QTensor.h"
#include "torch_musa/csrc/aten/quantized/TensorFactories.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

// TODO(@songlin.li): replace GeluQuantized with a int8 musa implementation
Tensor GeluQuantized(const Tensor& qx, c10::string_view approximate) {
  const OptionalDeviceGuard device_guard(device_of(qx));
  (void)approximate; // suppress unused variable lint warning
  if (qx.numel() == 0) {
    return Tensor{};
  }
  auto x_fp32 = at::dequantize(qx);
  auto result_fp32 = at::gelu(x_fp32);
  return at::musa::QuantizePerTensor(
      result_fp32, qx.q_scale(), qx.q_zero_point(), qx.scalar_type());
}

Tensor ReluQuantized(const Tensor& self) {
  const OptionalDeviceGuard device_guard(device_of(self));
  auto zero_point = self.q_zero_point();
  auto int_repr = self.int_repr();
  auto mask = (int_repr > zero_point);
  const auto relu_int_repr = at::where(mask, int_repr, zero_point);
  return at::musa::MakePerTensorQuantizedTensor(
      relu_int_repr, self.q_scale(), zero_point);
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("relu", TORCH_FN(ReluQuantized));
  m.impl("gelu", TORCH_FN(GeluQuantized));
  m.impl("relu_", TORCH_FN(at::native::relu_quantized_cuda_));
}

} // namespace musa
} // namespace at
