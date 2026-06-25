#pragma once

// RNG grid sizing overrides for Philox RNG alignment with NVIDIA GPUs.
// Reads TORCH_ARCH_MP_COUNT and TORCH_ARCH_MAXTHREADS_PER_MP environment
// variables to override multiProcessorCount and maxThreadsPerMultiProcessor
// used ONLY in RNG grid sizing (calc_execution_policy, Dropout Philox grid).

#include <cstdint>
#include <cstdlib>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"

namespace at {
namespace musa {

inline uint32_t getRNGMultiProcessorCount() {
  static const int32_t override_val = []() -> int32_t {
    const char* env = std::getenv("TORCH_ARCH_MP_COUNT");
    if (env) {
      int val = std::atoi(env);
      if (val > 0)
        return val;
    }
    return -1;
  }();
  if (override_val > 0)
    return static_cast<uint32_t>(override_val);
  return static_cast<uint32_t>(
      getCurrentDeviceProperties()->multiProcessorCount);
}

inline uint32_t getRNGMaxThreadsPerMultiProcessor() {
  static const int32_t override_val = []() -> int32_t {
    const char* env = std::getenv("TORCH_ARCH_MAXTHREADS_PER_MP");
    if (env) {
      int val = std::atoi(env);
      if (val > 0)
        return val;
    }
    return -1;
  }();
  if (override_val > 0)
    return static_cast<uint32_t>(override_val);
  return static_cast<uint32_t>(
      getCurrentDeviceProperties()->maxThreadsPerMultiProcessor);
}

} // namespace musa
} // namespace at
