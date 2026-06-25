#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_INTERPOLATE_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_INTERPOLATE_H_

#include <initializer_list>
#include <vector>

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

#define MUDNN_INTERPOLATE_APPLY(_, MODE) _(MODE, MUDNN_INTERPOLATE_##MODE)

#define MUDNN_FORALL_INTERPOLATE_OPS(_)     \
  MUDNN_INTERPOLATE_APPLY(_, NEAREST)       \
  MUDNN_INTERPOLATE_APPLY(_, NEAREST_EXACT) \
  MUDNN_INTERPOLATE_APPLY(_, LINEAR)        \
  MUDNN_INTERPOLATE_APPLY(_, BICUBIC)

class Interpolate : public Descriptor<
                        mudnnInterpolateStruct,
                        mudnnCreateInterpolateDescriptor,
                        mudnnDestroyInterpolateDescriptor> {
 public:
  enum class Mode {
#define CASE(MODE, _2) MODE,
    MUDNN_FORALL_INTERPOLATE_OPS(CASE)
#undef CASE
  };

  mudnnStatus_t Run(muHandle& h, muTensor& out, const muTensor& in) const;
  mudnnStatus_t RunBackward(
      muHandle& h,
      muTensor& grad_in,
      const muTensor& grad_out) const;
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::Interpolate;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::Interpolate;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetInterpolate(
    Interpolate& op,
    Interpolate::Mode mode,
    bool antialias,
    bool align_corners,
    const std::vector<float>& scales);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_INTERPOLATE_H_
