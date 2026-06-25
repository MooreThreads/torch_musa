#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_SCATTER_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_SCATTER_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

#define MUDNN_SCATTER_APPLY(_, MODE) _(MODE, MUDNN_SCATTER_MODE_##MODE)

#define MUDNN_FORALL_SCATTER_OPS(_)   \
  MUDNN_SCATTER_APPLY(_, UPDATE_ONLY) \
  MUDNN_SCATTER_APPLY(_, ADD)         \
  MUDNN_SCATTER_APPLY(_, SUB)         \
  MUDNN_SCATTER_APPLY(_, MUL)

class MudnnScatter : public Descriptor<
                         mudnnScatterStruct,
                         mudnnCreateScatterDescriptor,
                         mudnnDestroyScatterDescriptor> {
 public:
  enum class Mode {
#define CASE(MODE, _2) MODE,
    MUDNN_FORALL_SCATTER_OPS(CASE)
#undef CASE
  };

  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      const muTensor& idx,
      const muTensor& src,
      int dim,
      const MemoryMaintainer& maintainer) const;
};

} // namespace at::musa

namespace musa::dnn {
typedef class ::at::musa::MudnnScatter Scatter;
} // namespace musa::dnn

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetScatter(::musa::dnn::Scatter& op, ::musa::dnn::Scatter::Mode mode);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_SCATTER_H_
