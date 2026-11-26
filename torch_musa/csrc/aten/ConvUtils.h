#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/native/ConvUtils.h>
#include <c10/core/MemoryFormat.h>

namespace at {
namespace musa {
static inline at::MemoryFormat musa_conv_memory_format(
    const at::Tensor& input,
    const at::Tensor& weight) {
  // For 1x1 (2D) or 1x1x1 (3D) kernels, weight has spatial size 1, so
  // weight.suggest_memory_format() is ambiguous and always returns Contiguous
  // even after weight.to(memory_format=ChannelsLast/ChannelsLast3d).
  // Fall back to input.suggest_memory_format() to preserve channels-last
  // propagation across pointwise convolutions (matching upstream PyTorch).
  const auto weight_memory_format = weight.suggest_memory_format();
  if (weight_memory_format == at::MemoryFormat::ChannelsLast ||
      weight_memory_format == at::MemoryFormat::ChannelsLast3d) {
    return weight_memory_format;
  }
  const auto input_memory_format = input.suggest_memory_format();
  if (input_memory_format == at::MemoryFormat::ChannelsLast ||
      input_memory_format == at::MemoryFormat::ChannelsLast3d) {
    return input_memory_format;
  }
  return at::MemoryFormat::Contiguous;
}

static inline bool musa_conv_use_channels_last(
    const at::Tensor& input,
    const at::Tensor& weight) {
  if (!input.is_privateuseone() || !weight.is_privateuseone()) {
    return false;
  }

  const auto mf = musa_conv_memory_format(input, weight);
  return mf == at::MemoryFormat::ChannelsLast ||
      mf == at::MemoryFormat::ChannelsLast3d;
}
} // namespace musa
} // namespace at
