#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_POOLING_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_POOLING_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

#define MUDNN_FORALL_POOLING_OPS(_)                                         \
  _(MAXPOOL, MUDNN_POOLING_MAX)                                             \
  _(AVGPOOL_COUNT_PAD, MUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)         \
  _(AVGPOOL_COUNT_WITHOUT_PAD, MUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) \
  _(GLOBAL_AVGPOOL, MUDNN_POOLING_GLOBAL_AVGPOOL)                           \
  _(ADAPTIVE_MAXPOOL, MUDNN_POOLING_ADAPTIVE_MAXPOOL)                       \
  _(ADAPTIVE_AVGPOOL, MUDNN_POOLING_ADAPTIVE_AVGPOOL)

class Pooling : public Descriptor<
                    mudnnPoolingStruct,
                    mudnnCreatePoolingDescriptor,
                    mudnnDestroyPoolingDescriptor> {
 public:
  enum class Mode {
#define CASE(MODE, _2) MODE,
    MUDNN_FORALL_POOLING_OPS(CASE)
#undef CASE
  };

  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      const muTensor& in,
      muTensor& indices) const;

  mudnnStatus_t RunBwd(
      muHandle& h,
      muTensor& out,
      const muTensor& in,
      muTensor& indices) const;
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::Pooling;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::Pooling;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetPooling(Pooling& op, Pooling::Mode mode);

void SetPooling(
    Pooling& op,
    Pooling::Mode mode,
    std::initializer_list<int> kernel,
    std::initializer_list<int> pad,
    std::initializer_list<int> stride,
    std::initializer_list<int> dilation);

void SetPooling(
    Pooling& op,
    Pooling::Mode mode,
    std::initializer_list<int> kernel,
    std::initializer_list<int> pad,
    std::initializer_list<int> stride,
    std::initializer_list<int> dilation,
    int divisor);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_POOLING_H_
