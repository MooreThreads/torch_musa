#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch_musa/csrc/core/MUSAStream.h>

namespace torch::inductor {

// NOTICE: Following APIs are subject to change due to active development
// We provide NO BC guarantee for these APIs
class TORCH_API AOTIModelContainerRunnerMusa : public AOTIModelContainerRunner {
 public:
  // @param device_str: musa device string, e.g. "musa", "musa:0"
  AOTIModelContainerRunnerMusa(
      const std::string& model_so_path,
      size_t num_models = 1,
      const std::string& device_str = "musa",
      const std::string& mubin_dir = "");

  ~AOTIModelContainerRunnerMusa();

  std::vector<at::Tensor> run(std::vector<at::Tensor>& inputs);

  std::vector<at::Tensor> run_with_musa_stream(
      std::vector<at::Tensor>& inputs,
      c10::musa::MUSAStream musa_stream);
};

} // namespace torch::inductor
#endif
