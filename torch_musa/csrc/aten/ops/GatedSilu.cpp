#include <ATen/ATen.h>
#include <torch/library.h>

namespace at {
namespace musa {

at::Tensor GatedSilu(const at::Tensor& input);

} // namespace musa
} // namespace at