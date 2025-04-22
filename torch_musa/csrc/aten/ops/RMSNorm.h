#ifndef TORCH_MUSA_CSRC_ATEN_OPS_RMSNORM_H_
#define TORCH_MUSA_CSRC_ATEN_OPS_RMSNORM_H_

#include <ATen/ops/torch__fused_rmsnorm_forward_native.h>

namespace at::musa {

TORCH_API std::tuple<Tensor&, Tensor> FusedRMSNormForwardOut(
    const Tensor& input,
    Tensor& output,
    IntArrayRef normalized_shape,
    double eps = 1e-05,
    const c10::optional<Tensor>& weight_opt = {});

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_OPS_RMSNORM_H_
