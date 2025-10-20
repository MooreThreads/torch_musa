#include <torch/script.h>
#include <torch_musa/csrc/core/Device.h>
#include <cassert>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  // Register 'musa' for PrivateUse1 as we save model with 'musa'.
  c10::register_privateuse1_backend("musa");

  torch::jit::script::Module module;
  try {
    // Load model which saved with torch jit.trace or jit.script.
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::vector<torch::jit::IValue> inputs;
  // Ready for input data.
  torch::Tensor input = torch::rand({1, 3, 224, 224}).to("musa");
  assert(input.is_privateuseone() == input.is_musa());
  assert(input.device().is_privateuseone() == input.device().is_musa());
  inputs.push_back(input);

  // Model execute.
  at::Tensor output = module.forward(inputs).toTensor();

  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;

  return 0;
}
