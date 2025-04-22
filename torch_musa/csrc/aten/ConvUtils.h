#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/native/ConvUtils.h>
#include <c10/core/MemoryFormat.h>

namespace at {
namespace musa {
static inline bool musa_conv_use_channels_last(
    const at::Tensor& input,
    const at::Tensor& weight) {
  if (!input.is_privateuseone() || !weight.is_privateuseone()) {
    return false;
  }

  // Determine whether to use channels_last based on weight's memory format
  auto weight_memory_format = weight.suggest_memory_format();

  bool can_use_musa_channels_last_2d =
      weight_memory_format == at::MemoryFormat::ChannelsLast;
  bool can_use_musa_channels_last_3d =
      weight_memory_format == at::MemoryFormat::ChannelsLast3d;

  return can_use_musa_channels_last_2d || can_use_musa_channels_last_3d;
}
} // namespace musa
} // namespace at
