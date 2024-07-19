#ifndef TORCH_MUSA_CSRC_ATEN_OPS_RANDPERM_H_
#define TORCH_MUSA_CSRC_ATEN_OPS_RANDPERM_H_

namespace at {
namespace musa {

Tensor& RandpermOutMusa(
    int64_t n,
    c10::optional<Generator> generator,
    Tensor& result);

} // namespace musa
} // namespace at

#endif // TORCH_MUSA_CSRC_ATEN_OPS_RANDPERM_H_
