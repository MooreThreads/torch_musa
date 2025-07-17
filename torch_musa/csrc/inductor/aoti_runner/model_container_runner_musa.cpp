#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch_musa/csrc/inductor/aoti_runner/model_container_runner_musa.h>

namespace torch::inductor {

AOTIModelContainerRunnerMusa::AOTIModelContainerRunnerMusa(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& mubin_dir)
    : AOTIModelContainerRunner(
          model_so_path,
          num_models,
          device_str,
          mubin_dir) {}

AOTIModelContainerRunnerMusa::~AOTIModelContainerRunnerMusa() = default;

std::vector<at::Tensor> AOTIModelContainerRunnerMusa::run(
    std::vector<at::Tensor>& inputs) {
  c10::musa::MUSAStream musa_stream = c10::musa::getCurrentMUSAStream();
  return AOTIModelContainerRunner::run(
      inputs, reinterpret_cast<AOTInductorStreamHandle>(musa_stream.stream()));
}

std::vector<at::Tensor> AOTIModelContainerRunnerMusa::run_with_musa_stream(
    std::vector<at::Tensor>& inputs,
    c10::musa::MUSAStream musa_stream) {
  return AOTIModelContainerRunner::run(
      inputs, reinterpret_cast<AOTInductorStreamHandle>(musa_stream.stream()));
}

} // namespace torch::inductor
#endif
