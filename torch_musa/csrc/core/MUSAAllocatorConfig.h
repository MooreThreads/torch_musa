#pragma once

#include <c10/musa/MUSAMacros.h>
#include <c10/util/Exception.h>
#include <c10/util/llvmMathExtras.h>
// #include <cuda_runtime_api.h>

#include "torch_musa/csrc/core/MUSACachingAllocator.h"
#include "torch_musa/csrc/core/MUSAException.h"

#include <atomic>
#include <vector>

namespace c10 {
namespace musa {
namespace MUSACachingAllocator {

// Environment config parser
class C10_MUSA_API MUSAAllocatorConfig {
 public:
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }
  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

  static bool expandable_segments() {
#if !defined(REAL_MUSA_VERSION) || REAL_MUSA_VERSION < 4000
    if (instance().m_expandable_segments) {
      TORCH_WARN_ONCE("expandable_segments not supported on this platform")
    }
    return false;
#else
    return instance().m_expandable_segments;
#endif
  }

  static bool release_lock_on_musamalloc() {
    return instance().m_release_lock_on_musamalloc;
  }

  /** Pinned memory allocator settings */
  static bool pinned_use_musa_host_register() {
    return instance().m_pinned_use_musa_host_register;
  }

  static size_t pinned_num_register_threads() {
    return instance().m_pinned_num_register_threads;
  }

  static size_t pinned_max_register_threads() {
    // Based on the benchmark results, we see better allocation performance
    // with 8 threads. However on future systems, we may need more threads
    // and limiting this to 128 threads.
    return 128;
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As ane example, if we want 4 divisions between 2's power, this can be done
  // using env variable: PYTORCH_MUSA_ALLOC_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions(size_t size);

  static MUSAAllocatorConfig& instance() {
    static MUSAAllocatorConfig* s_instance = ([]() {
      auto inst = new MUSAAllocatorConfig();
      const char* env = getenv("PYTORCH_MUSA_ALLOC_CONF");
      inst->parseArgs(env);
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const char* env);

 private:
  MUSAAllocatorConfig();

  void lexArgs(const char* env, std::vector<std::string>& config);
  void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);
  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);
  size_t parseGarbageCollectionThreshold(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseRoundUpPower2Divisions(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseAllocatorConfig(
      const std::vector<std::string>& config,
      size_t i,
      bool& used_musaMallocAsync);
  size_t parsePinnedUseMusaHostRegister(
      const std::vector<std::string>& config,
      size_t i);
  size_t parsePinnedNumRegisterThreads(
      const std::vector<std::string>& config,
      size_t i);

  std::atomic<size_t> m_max_split_size;
  std::vector<size_t> m_roundup_power2_divisions;
  std::atomic<double> m_garbage_collection_threshold;
  std::atomic<size_t> m_pinned_num_register_threads;
  std::atomic<bool> m_expandable_segments;
  std::atomic<bool> m_release_lock_on_musamalloc;
  std::atomic<bool> m_pinned_use_musa_host_register;
};

// General caching allocator utilities
C10_MUSA_API void setAllocatorSettings(const std::string& env);

} // namespace MUSACachingAllocator
} // namespace musa
} // namespace c10
