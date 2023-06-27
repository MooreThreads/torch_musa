#pragma once
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/Tensor.h>

#include <mudnn.h>

namespace at {
namespace musa {
void EmbeddingRun(
    const Tensor& o,
    const Tensor& t,
    const Tensor& i,
    int64_t padding);
} // namespace musa
} // namespace at