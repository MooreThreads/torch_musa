#pragma once
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/Tensor.h>

namespace at {
namespace musa {

Tensor& AmpUpdateScaleMusa(
    Tensor& current_scale,
    Tensor& growth_tracker,
    const Tensor& found_inf,
    float growth_factor,
    float backoff_factor,
    int64_t growth_interval);
} // namespace musa

} // namespace at
