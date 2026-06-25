#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_DROPOUT_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_DROPOUT_H_

#include <cstdint>

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class Dropout : public Descriptor<
                    mudnnDropoutStruct,
                    mudnnCreateDropoutDescriptor,
                    mudnnDestroyDropoutDescriptor> {
 public:
  mudnnStatus_t SetP(double p);
  mudnnStatus_t SetScale(double scale);

  mudnnStatus_t SetSeed(uint64_t seed);
  mudnnStatus_t SetSeed(uint64_t* seed_ptr);

  mudnnStatus_t SetOffset(uint64_t offset);
  mudnnStatus_t SetOffset(uint64_t* offset_ptr, uint64_t sub_offset);

  mudnnStatus_t RunDropout(
      muHandle& h,
      muTensor& out,
      const muTensor& in,
      muTensor& mask);

  mudnnStatus_t RunDropoutBwd(
      muHandle& h,
      muTensor& out,
      const muTensor& grad,
      const muTensor& mask);

 private:
  void BuildForwardDescriptor();
  void BuildBackwardDescriptor();

  double p_ = 0.0;
  double scale_ = 1.0;

  mudnnParamDataType_t seed_type_ = MUDNN_PARAM_DATA_NONE;
  uint64_t seed_val_ = 0;
  uint64_t* seed_ptr_ = nullptr;

  mudnnParamDataType_t offset_type_ = MUDNN_PARAM_DATA_NONE;
  uint64_t offset_val_ = 0;
  uint64_t* offset_ptr_ = nullptr;
  uint64_t sub_offset_ = 0;
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::Dropout;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::Dropout;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_DROPOUT_H_
