#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch_musa/csrc/inductor/aoti_runner/model_container_runner_musa.h>

namespace torch::inductor {

AOTIModelContainerRunnerMusa::AOTIModelContainerRunnerMusa(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& mubin_dir,
    const bool run_single_threaded)
    : AOTIModelContainerRunner(
          model_so_path,
          num_models,
          device_str,
          mubin_dir,
          run_single_threaded) {}

AOTIModelContainerRunnerMusa::~AOTIModelContainerRunnerMusa() = default;

std::vector<at::Tensor> AOTIModelContainerRunnerMusa::run_impl(
    std::vector<AtenTensorHandle>& input_handles,
    void* stream_handle) {
  if (stream_handle == nullptr) {
    c10::musa::MUSAStream musa_stream = c10::musa::getCurrentMUSAStream();
    stream_handle = reinterpret_cast<void*>(musa_stream.stream());
  }
  return AOTIModelContainerRunner::run_impl(input_handles, stream_handle);
}

std::vector<at::Tensor> AOTIModelContainerRunnerMusa::run_with_musa_stream(
    const std::vector<at::Tensor>& inputs,
    const c10::musa::MUSAStream& musa_stream) {
  return run(inputs, reinterpret_cast<void*>(musa_stream.stream()));
}

namespace {
std::unique_ptr<AOTIModelContainerRunner> create_aoti_runner_musa(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& mubin_dir,
    const bool run_single_threaded) {
  return std::make_unique<AOTIModelContainerRunnerMusa>(
      model_so_path, num_models, device_str, mubin_dir, run_single_threaded);
}
} // namespace

RegisterAOTIModelRunner register_musa_runner("musa", &create_aoti_runner_musa);

} // namespace torch::inductor
#endif
