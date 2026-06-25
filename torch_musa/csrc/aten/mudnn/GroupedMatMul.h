#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_GROUPEDMATMUL_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_GROUPEDMATMUL_H_

#include "torch_musa/csrc/aten/mudnn/BatchMatMul.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"

namespace at::musa {

class GroupedMatMul : public Descriptor<
                          mudnnGroupedMatMulStruct,
                          mudnnCreateGroupedMatMulDescriptor,
                          mudnnDestroyGroupedMatMulDescriptor> {
 public:
  mudnnStatus_t RunLt(
      muHandle& h,
      muTensor* d,
      const muTensor* a,
      const muTensor* b,
      const muTensor* c,
      const muTensor* bias,
      const MatMulLtParam* params,
      int groups,
      const MemoryMaintainer& maintainer) const;
};

} // namespace at::musa

namespace musa::dnn {
typedef at::musa::GroupedMatMul GroupedMatMul;
} // namespace musa::dnn

#else

namespace at::musa {
typedef ::musa::dnn::GroupedMatMul GroupedMatMul;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetGroupedMatMul(
    GroupedMatMul& op,
    ComputeMode mode,
    bool transa,
    bool transb,
    double alpha,
    double beta,
    double gamma);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_GROUPEDMATMUL_H_
