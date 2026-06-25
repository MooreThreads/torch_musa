#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_PAD_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_PAD_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

#define MUDNN_PAD_APPLY(_, MODE) _(MODE, MUDNN_PAD_MODE_##MODE)

#define MUDNN_FORALL_PAD_OPS(_) \
  MUDNN_PAD_APPLY(_, CONSTANT)  \
  MUDNN_PAD_APPLY(_, REFLECT)   \
  MUDNN_PAD_APPLY(_, REPLICATE) \
  MUDNN_PAD_APPLY(_, CIRCULAR)

class Pad : public Descriptor<
                mudnnPadStruct,
                mudnnCreatePadDescriptor,
                mudnnDestroyPadDescriptor> {
 public:
  enum class Mode {
#define CASE(MODE, _2) MODE,
    MUDNN_FORALL_PAD_OPS(CASE)
#undef CASE
  };

  mudnnStatus_t Run(muHandle& h, muTensor& out, const muTensor& in) const;
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::Pad;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::Pad;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetPad(Pad& op, Pad::Mode mode, int len, const int* pad);

void SetPad(Pad& op, Pad::Mode mode, double v, int len, const int* pad);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_PAD_H_
