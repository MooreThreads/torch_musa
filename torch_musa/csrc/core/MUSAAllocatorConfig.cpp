#include "torch_musa/csrc/core/MUSAAllocatorConfig.h"

#include <c10/util/llvmMathExtras.h>

#include "torch_musa/csrc/core/MUSACachingAllocator.h"

namespace c10::musa::MUSACachingAllocator {

size_t MUSAAllocatorConfig::parseAllocatorConfig(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i,
    bool& used_musaMallocAsync) {
  tokenizer.checkToken(++i, ":");
  i++; // Move to the value after the colon
  TORCH_CHECK_VALUE(
      ((tokenizer[i] == "native") || (tokenizer[i] == "musaMallocAsync") ||
       (tokenizer[i] == "unified")),
      "Unknown allocator backend, "
      "options are native and musaMallocAsync and unified");
  used_musaMallocAsync = (tokenizer[i] == "musaMallocAsync");
  if (used_musaMallocAsync) {
#if !defined(REAL_MUSA_VERSION) || (REAL_MUSA_VERSION < 5010)
    TORCH_CHECK(
        false,
        "backend:musaMallocAsync is ON, but not supported for MUSA yet. "
        "MUSA SDK >= 5.1 is required.");
#endif
  }
  if (tokenizer[i] == "unified") {
    TORCH_INTERNAL_ASSERT(
        get()->name() == "pluggable",
        "expected allocator backend parsed at runtime for unified to be "
        "pluggable, but got ",
        get()->name());
  } else {
    TORCH_INTERNAL_ASSERT(
        tokenizer[i] == get()->name(),
        "Allocator backend parsed at runtime != "
        "allocator backend parsed at load time");
  }
  return i;
}

void MUSAAllocatorConfig::parseArgs(const std::string& env) {
  bool used_musaMallocAsync = false;
  bool used_native_specific_option = false;

  c10::CachingAllocator::ConfigTokenizer tokenizer(env);
  for (size_t i = 0; i < tokenizer.size(); i++) {
    const auto& key = tokenizer[i];
    if (key == "backend") {
      i = parseAllocatorConfig(tokenizer, i, used_musaMallocAsync);
    } else if (key == "release_lock_on_musamalloc") {
      used_native_specific_option = true;
      tokenizer.checkToken(++i, ":");
      m_release_lock_on_musamalloc = tokenizer.toBool(++i);
    } else if (key == "pinned_use_musa_host_register") {
      i = parsePinnedUseMusaHostRegister(tokenizer, i);
      used_native_specific_option = true;
    } else if (key == "pinned_num_register_threads") {
      i = parsePinnedNumRegisterThreads(tokenizer, i);
      used_native_specific_option = true;
    } else if (key == "pinned_reserve_segment_size_mb") {
      i = parsePinnedReserveSegmentSize(tokenizer, i);
      used_native_specific_option = true;
    } else if (key == "graph_capture_record_stream_reuse") {
      i = parseGraphCaptureRecordStreamReuse(tokenizer, i);
      used_native_specific_option = true;
    } else if (key == "per_process_memory_fraction") {
      i = parsePerProcessMemoryFraction(tokenizer, i);
      used_native_specific_option = true;
    } else {
      const auto& keys =
          c10::CachingAllocator::AcceleratorAllocatorConfig::getKeys();
      TORCH_CHECK_VALUE(
          keys.find(key) != keys.end(),
          "Unrecognized key '",
          key,
          "' in MUSA allocator config.");
      // Skip the key and its value
      i = tokenizer.skipKey(i);
    }

    if (i + 1 < tokenizer.size()) {
      tokenizer.checkToken(++i, ",");
    }
  }

  if (used_musaMallocAsync && used_native_specific_option) {
    TORCH_WARN(
        "backend:musaMallocAsync ignores max_split_size_mb,"
        "roundup_power2_divisions, and garbage_collect_threshold.");
  }
}

size_t MUSAAllocatorConfig::parsePinnedUseMusaHostRegister(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  m_pinned_use_musa_host_register = tokenizer.toBool(++i);
  return i;
}

size_t MUSAAllocatorConfig::parseGraphCaptureRecordStreamReuse(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  m_graph_capture_record_stream_reuse = tokenizer.toBool(++i);
  return i;
}

double MUSAAllocatorConfig::parsePerProcessMemoryFraction(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  double val_env = tokenizer.toDouble(++i);
  TORCH_CHECK_VALUE(
      val_env >= 0.0 && val_env <= 1.0,
      "per_process_memory_fraction is invalid, set it in [0.0, 1.0]");
  m_per_process_memory_fraction = val_env;
  return i;
}

size_t MUSAAllocatorConfig::parsePinnedNumRegisterThreads(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  size_t val2 = tokenizer.toSizeT(++i);
  TORCH_CHECK_VALUE(
      llvm::isPowerOf2_64(val2),
      "Number of register threads has to be power of 2, got ",
      val2);
  auto maxThreads = MUSAAllocatorConfig::pinned_max_register_threads();
  TORCH_CHECK_VALUE(
      val2 <= maxThreads,
      "Number of register threads should be less than or equal to ",
      maxThreads,
      ", got ",
      val2);
  m_pinned_num_register_threads = val2;
  return i;
}

size_t MUSAAllocatorConfig::parsePinnedReserveSegmentSize(
    const c10::CachingAllocator::ConfigTokenizer& tokenizer,
    size_t i) {
  tokenizer.checkToken(++i, ":");
  size_t val2 = tokenizer.toSizeT(++i);
  TORCH_CHECK_VALUE(
      val2 > 0, "Pinned reserve segment size has to be greater than 0");
  m_pinned_reserve_segment_size_mb = val2;
  return i;
}

REGISTER_ALLOCATOR_CONFIG_PARSE_HOOK(MUSAAllocatorConfig)

} // namespace c10::musa::MUSACachingAllocator
