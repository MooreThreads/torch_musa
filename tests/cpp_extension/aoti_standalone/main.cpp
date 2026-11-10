#include <ATen/Functions.h>
#include <c10/core/DeviceType.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch_musa/csrc/core/MUSAHooksInterface.h>

#include <fstream>
#include <iostream>
#include <string>

namespace {

void check_no_python_libraries() {
  std::ifstream maps("/proc/self/maps");
  TORCH_CHECK(maps.is_open(), "Unable to inspect process mappings");
  std::string line;
  while (std::getline(maps, line)) {
    TORCH_CHECK(
        line.find("libpython") == std::string::npos &&
            line.find("libtorch_python") == std::string::npos &&
            line.find("libmusa_python") == std::string::npos &&
            line.find("_MUSAC") == std::string::npos,
        "Python component loaded in the standalone process: ",
        line);
  }
}

} // namespace

int main(int argc, char** argv) {
  TORCH_CHECK(argc == 2, "Usage: aoti_standalone model.pt2");
  c10::InferenceMode guard;
  c10::register_privateuse1_backend("musa");
  at::detail::getMUSAHooks();
  check_no_python_libraries();

  auto cpu_input = at::div(
      at::arange(16, at::TensorOptions().dtype(at::kFloat)).reshape({4, 4}),
      16.0);
  auto expected = at::relu(at::sub(at::mm(cpu_input, cpu_input), 0.5));
  auto input = cpu_input.to(c10::Device(c10::DeviceType::PrivateUse1, 0));
  torch::inductor::AOTIModelPackageLoader loader(argv[1], "model", true, 1, 0);
  for (int i = 0; i < 3; ++i) {
    auto outputs = loader.run({input});
    TORCH_CHECK(outputs.size() == 1, "Unexpected number of model outputs");
    TORCH_CHECK(outputs[0].device().type() == c10::DeviceType::PrivateUse1);
    TORCH_CHECK(
        at::allclose(outputs[0].cpu(), expected, 1e-4, 1e-4),
        "Standalone AOTI output differs from the CPU reference");
  }
  check_no_python_libraries();
  std::cout << "PASS: standalone MUSA AOTI matches CPU; no Python libraries\n";
}
