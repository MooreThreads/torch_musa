#include <torch/extension.h>

extern std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_musa(
    int chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists,
    at::optional<bool> per_tensor_python);

extern void multi_tensor_lamb_musa(
    int chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int step,
    const int bias_correction,
    const float weight_decay,
    const int grad_averaging,
    const int mode,
    at::Tensor global_grad_norm,
    const float max_grad_norm,
    at::optional<bool> use_nvlamb_python);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "multi_tensor_l2norm",
      &multi_tensor_l2norm_musa,
      "Computes L2 norm for a list of contiguous tensors");
  m.def(
      "multi_tensor_lamb",
      &multi_tensor_lamb_musa,
      "Computes and apply update for LAMB optimizer");
}
