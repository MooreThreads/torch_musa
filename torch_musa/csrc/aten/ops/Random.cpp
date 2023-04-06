#pragma GCC diagnostic push

#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#pragma GCC diagnostic pop

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace native {
namespace musa {
/// from native/TensorFactories.cpp
at::Tensor Randint(
    int64_t high,
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  UNUSED(device);
  auto a1 = at::randint(
      high,
      size,
      c10::nullopt /* generator*/,
      dtype,
      layout,
      Device::Type::CPU,
      pin_memory);
  a1 = a1.to("musa");
  return a1;
}

at::Tensor& RandomFrom(
    at::Tensor& self,
    int64_t from,
    c10::optional<int64_t> to,
    c10::optional<at::Generator> generator) {
  Tensor& a1 = at::native::random_(self, from, to, generator);
  a1 = a1.to("musa");
  return a1;
}

at::Tensor GeneratorRandint(
    int64_t high,
    at::IntArrayRef size,
    c10::optional<at::Generator> generator,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  // DeviceGuard omitted
  UNUSED(device);
  auto a1 = at::native::randint(
      high, size, generator, dtype, layout, Device::Type::CPU, pin_memory);
  a1 = a1.to("musa");
  return a1;
}

at::Tensor LowRandint(
    int64_t low,
    int64_t high,
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  // DeviceGuard omitted
  UNUSED(device);
  auto a1 = at::native::randint(
      low, high, size, dtype, layout, Device::Type::CPU, pin_memory);
  a1 = a1.to("musa");
  return a1;
}

at::Tensor LowGeneratorRandint(
    int64_t low,
    int64_t high,
    at::IntArrayRef size,
    c10::optional<at::Generator> generator,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  // DeviceGuard omitted
  UNUSED(device);
  auto a1 = at::native::randint(
      low, high, size, generator, dtype, layout, Device::Type::CPU, pin_memory);
  a1 = a1.to("musa");
  return a1;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("randint", &Randint);
  m.impl("randint.generator", &GeneratorRandint);
  m.impl("randint.low", &LowRandint);
  m.impl("randint.low_generator", &LowGeneratorRandint);
  m.impl("random_.from", &RandomFrom);
}

} // namespace musa
} // namespace native
} // namespace at
