#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorCompare.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/quantized/TensorFactories.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
namespace musa {

std::tuple<Tensor, Tensor> QMax(const Tensor& self, int64_t dim, bool keepdim) {
  c10::musa::MUSAGuard device_guard(self.device());
  TORCH_CHECK(
      self.qscheme() == at::kPerTensorAffine,
      "Max operator for quantized tensors only works for per tensor quantized tensors. "
      "Please open an issue on https://github.com/pytorch/pytorch/issues if you need per channel quantized tensor support.");
  Tensor max_indices = at::empty({0}, self.options().dtype(kLong));
  Tensor max =
      at::empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));
  at::max_outf(self.int_repr(), dim, keepdim, max, max_indices);
  return std::tuple<Tensor, Tensor>(
      at::musa::MakePerTensorQuantizedTensor(
          max, self.q_scale(), self.q_zero_point()),
      max_indices);
}

std::tuple<Tensor, Tensor> QMin(const Tensor& self, int64_t dim, bool keepdim) {
  c10::musa::MUSAGuard device_guard(self.device());
  TORCH_CHECK(
      self.qscheme() == at::kPerTensorAffine,
      "Min operator for quantized tensors only works for per tensor quantized tensors. "
      "Please open an issue on https://github.com/pytorch/pytorch/issues if you need per channel quantized tensor support.");
  Tensor min_indices = at::empty({0}, self.options().dtype(kLong));
  Tensor min =
      at::empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));
  at::min_outf(self.int_repr(), dim, keepdim, min, min_indices);
  return std::tuple<Tensor, Tensor>(
      at::musa::MakePerTensorQuantizedTensor(
          min, self.q_scale(), self.q_zero_point()),
      min_indices);
}

} // namespace musa
} // namespace at
