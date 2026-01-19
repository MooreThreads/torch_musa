#ifndef TORCH_MUSA_CSRC_ATEN_OPS_COPY_H_
#define TORCH_MUSA_CSRC_ATEN_OPS_COPY_H_

namespace at {
struct TensorIteratorBase;

namespace musa {

void direct_copy_kernel_musa(TensorIteratorBase& iter);

}
} // namespace at

#endif // TORCH_MUSA_CSRC_ATEN_OPS_COPY_H_
