#include <ATen/Context.h>
#include <c10/core/ScalarType.h>

#include "torch_musa/csrc/aten/utils/Context.h"

namespace at::musa {

Context& GlobalContext() {
  static Context global_context_;
  return global_context_;
}

bool Context::GetAllowTF32() const {
  return allow_tf32_;
}

void Context::SetAllowTF32(bool allow_tf32) {
  allow_tf32_ = allow_tf32;
}

bool Context::GetAllowMublasTF32() const {
  return allow_mublas_tf32_;
}

void Context::SetAllowMublasTF32(bool allow_tf32) {
  allow_mublas_tf32_ = allow_tf32;
}

static constexpr const auto mublas_config_var_name = "MUBLAS_WORKSPACE_CONFIG";
static constexpr const std::array<const char*, 2> mublas_deterministic_configs =
    {":4096:8", ":16:8"};

bool Context::checkMuBLASConfigDeterministic() {
  if (hasMUSART()) {
    const auto workspace_config = c10::utils::get_env(mublas_config_var_name);
    return (
        workspace_config == mublas_deterministic_configs[0] ||
        workspace_config == mublas_deterministic_configs[1]);
  }
  return true;
}

void Context::alertMuBLASConfigNotDeterministic() const {
  static const bool mublas_config_deterministic =
      checkMuBLASConfigDeterministic();
  if (C10_LIKELY(
          !at::globalContext().deterministicAlgorithms() ||
          mublas_config_deterministic)) {
    return;
  }

  auto msg = c10::str(
      "Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or ",
      "`at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because ",
      "it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this ",
      "case, you must set an environment variable before running your PyTorch application: ",
      mublas_config_var_name,
      "=",
      mublas_deterministic_configs[0],
      " or ",
      mublas_config_var_name,
      "=",
      mublas_deterministic_configs[1],
      ". For more information, go to ",
      "https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility");

  if (at::globalContext().deterministicAlgorithmsWarnOnly()) {
    TORCH_WARN(msg);
  } else {
    TORCH_CHECK(false, msg);
  }
}

} // namespace at::musa
