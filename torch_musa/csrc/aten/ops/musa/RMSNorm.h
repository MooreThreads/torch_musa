#ifndef TORCH_MUSA_CSRC_ATEN_OPS_MUSA_RMSNORM_H_
#define TORCH_MUSA_CSRC_ATEN_OPS_MUSA_RMSNORM_H_

namespace at {
namespace musa {

void musa_rms_norm(
    const at::Tensor& input,
    at::Tensor& invvar,
    at::Tensor& output,
    at::Tensor& gamma,
    int inner,
    int outter,
    at::IntArrayRef normalized_shape,
    double epsilon);

void musa_rms_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& invvar,
    const at::Tensor& input,
    const at::Tensor& gamma,
    at::Tensor& grad_input,
    at::Tensor& grad_gamma,
    at::IntArrayRef normalized_shape,
    int n1,
    int n2,
    double eps);

} // namespace musa
} // namespace at

#endif // TORCH_MUSA_CSRC_ATEN_OPS_MUSA_RMSNORM_H_
