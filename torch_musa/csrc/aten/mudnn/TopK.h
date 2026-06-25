#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_TOPK_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_TOPK_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class TopK : public Descriptor<
                 mudnnTopKStruct,
                 mudnnCreateTopKDescriptor,
                 mudnnDestroyTopKDescriptor> {
 public:
  mudnnStatus_t Run(
      muHandle& h,
      muTensor& values,
      muTensor& indices,
      const muTensor& input,
      const MemoryMaintainer& maintainer) const;
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::TopK;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::TopK;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetTopK(TopK& op, int64_t k, int64_t dim, bool largest, bool sorted);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_TOPK_H_
