#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_SORTBYKEY_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_SORTBYKEY_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class SortByKey : public Descriptor<
                      mudnnSortByKeyStruct,
                      mudnnCreateSortByKeyDescriptor,
                      mudnnDestroySortByKeyDescriptor> {
 public:
  mudnnStatus_t Run(
      muHandle& h,
      muTensor& keys_out,
      muTensor& values_out,
      const muTensor& keys_in,
      const muTensor& values_in,
      const MemoryMaintainer& maintainer) const;
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::SortByKey;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::SortByKey;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetSortByKey(SortByKey& op, int dim, bool stable, bool descending);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_SORTBYKEY_H_
