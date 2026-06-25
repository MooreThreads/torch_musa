#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_SORT_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_SORT_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class MudnnSort : public Descriptor<
                      mudnnSortStruct,
                      mudnnCreateSortDescriptor,
                      mudnnDestroySortDescriptor> {
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
typedef class ::at::musa::MudnnSort Sort;
} // namespace musa::dnn

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetSort(::musa::dnn::Sort& op, int64_t dim, bool descending, bool stable);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_SORT_H_
