#include <torch_musa/csrc/core/MCCL.h>

#include <c10/util/Exception.h>

#ifdef USE_MCCL
#include <mccl.h>
#endif

namespace torch::musa::mccl {

std::uint64_t version() {
#ifdef USE_MCCL
  constexpr std::uint64_t ver = (((uint64_t)MCCL_MAJOR) << 32) |
      (((uint64_t)MCCL_MINOR) << 16) | ((uint64_t)MCCL_PATCH);
  return ver;
#else
  return 0;
#endif
}

const char* version_suffix() {
#ifdef USE_MCCL
  return MCCL_SUFFIX;
#else
  AT_ERROR("torch_musa built without MCCL support");
#endif
}

} // namespace torch::musa::mccl
