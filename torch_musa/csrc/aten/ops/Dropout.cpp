#include <ATen/Config.h>
#include <ATen/ExpandUtils.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros_like.h>
#endif
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include "torch_musa/csrc/aten/musa/MUSAGraphsUtils.muh"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {
::std::tuple<Tensor, Tensor> NativeDropout(
    const Tensor& input,
    double p,
    c10::optional<bool> train) {
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of NativeDropout must be MUSA, ",
      "but now is ",
      input.device());

  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float ||
          input.scalar_type() == at::ScalarType::Half ||
          input.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of input tensor of NativeDropout only support Float, Half and BFloat16, ",
      "but now it is ",
      input.scalar_type());

  c10::musa::MUSAGuard device_guard(input.device());
  if (input.numel() == 0) {
    return std::make_tuple(input, at::empty_like(input, input.options()));
  }
  // short-cut for train == false
  if (train.has_value() && !train.value()) {
    return std::make_tuple(
        input.clone(),
        at::ones_like(
            input,
            input.options().dtype(c10::CppTypeToScalarType<bool>::value)));
  }
  // short-cut
  if (p == 1) {
    // native_dropout is in derivatives.yaml, so we don't need to add data
    // dependency from output to input for autograd
    auto ret = at::zeros_like(input);
    auto mask = at::zeros_like(
        input, input.options().dtype(c10::CppTypeToScalarType<bool>::value));
    return std::tuple<Tensor, Tensor>(ret, mask);
  }

  Tensor mask = at::empty_like(
      input,
      input.options().dtype(c10::CppTypeToScalarType<bool>::value),
      at::MemoryFormat::Contiguous);
  Tensor output = at::empty_like(input, at::MemoryFormat::Contiguous);
  muHandle& h = GetMudnnHandle();
  auto contiguous_input = input.contiguous();
  auto musa_input = CreateMUTensor(contiguous_input);
  auto musa_output = CreateMUTensor(output);
  auto musa_mask = CreateMUTensor(mask);
  const int UNROLL = 8;
  const int64_t nelem = input.numel();
  const int64_t block_size = 128;
  auto gen = get_generator_or_default<MUSAGeneratorImpl>(
      c10::nullopt, musa::detail::getDefaultMUSAGenerator());
  unsigned int blocks_per_sm =
      at::musa::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor /
      block_size;
  unsigned int g_x = (nelem + block_size - 1) / block_size;
  g_x = std::min(
      (unsigned int)at::musa::getCurrentDeviceProperties()
              ->multiProcessorCount *
          blocks_per_sm,
      g_x);
  // number of times random will be generated per thread, to offset philox
  // counter in thc random state
  int64_t counter_offset =
      ((nelem - 1) / (block_size * g_x * UNROLL) + 1) * UNROLL;
  PhiloxMusaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_musa_state(counter_offset);
  }
  auto seeds = at::musa::philox::unpack(rng_engine_inputs);
  ::musa::dnn::Dropout dropout;
  CHECK_MUDNN_STATUS(dropout.SetP(p), "SetP");
  CHECK_MUDNN_STATUS(dropout.SetSeed(std::get<0>(seeds)), "SetSeed");
  CHECK_MUDNN_STATUS(dropout.SetOffset(std::get<1>(seeds)), "SetOffset");
  CHECK_MUDNN_STATUS(
      dropout.RunDropout(h, musa_output, musa_input, musa_mask), "RunDropout");
  return std::make_tuple(output, mask);
}

Tensor NativeDropoutBackward(
    const Tensor& grad_output,
    const Tensor& mask,
    double scale) {
  TORCH_CHECK(
      grad_output.device().type() == kMUSA,
      "Device of input tensor of NativeDropoutBackward must be MUSA,",
      " but now is ",
      grad_output.device());
  TORCH_CHECK(
      grad_output.scalar_type() == at::ScalarType::Float ||
          grad_output.scalar_type() == at::ScalarType::Half ||
          grad_output.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of input tensor of NativeDropoutBackward only support",
      " Float, Half and BFloat16, but now it is ",
      grad_output.scalar_type());
  TORCH_CHECK(
      mask.device().type() == kMUSA,
      "Device of mask tensor of NativeDropoutBackward must be MUSA,",
      " but now is ",
      mask.device());
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Bool,
      "Dtype of mask tensor of NativeDropoutBackward only support",
      " Bool, but now it is ",
      mask.scalar_type());
  c10::musa::MUSAGuard device_guard(grad_output.device());
  Tensor output = at::empty_like(grad_output, at::MemoryFormat::Contiguous);
  muHandle& h = GetMudnnHandle();
  auto contiguous_grad_output = grad_output.contiguous();
  auto contiguous_mask = mask.contiguous();
  auto musa_grad_output = CreateMUTensor(contiguous_grad_output);
  auto musa_mask = CreateMUTensor(contiguous_mask);
  auto musa_output = CreateMUTensor(output);

  ::musa::dnn::Dropout dropout;
  dropout.SetScale(scale);
  CHECK_MUDNN_STATUS(
      dropout.RunDropoutBwd(h, musa_output, musa_grad_output, musa_mask),
      "RunDropoutBwd");
  return output;
}

} // namespace musa
} // namespace at
