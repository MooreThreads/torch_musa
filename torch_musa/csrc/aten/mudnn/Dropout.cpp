#include "torch_musa/csrc/aten/mudnn/Dropout.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t Dropout::SetP(double p) {
  p_ = p;
  return MUDNN_STATUS_SUCCESS;
}

mudnnStatus_t Dropout::SetScale(double scale) {
  scale_ = scale;
  return MUDNN_STATUS_SUCCESS;
}

mudnnStatus_t Dropout::SetSeed(uint64_t seed) {
  seed_type_ = MUDNN_PARAM_DATA_UINT64;
  seed_val_ = seed;
  seed_ptr_ = nullptr;
  return MUDNN_STATUS_SUCCESS;
}

mudnnStatus_t Dropout::SetSeed(uint64_t* seed_ptr) {
  seed_type_ = MUDNN_PARAM_DATA_POINTER;
  seed_ptr_ = seed_ptr;
  return MUDNN_STATUS_SUCCESS;
}

mudnnStatus_t Dropout::SetOffset(uint64_t offset) {
  offset_type_ = MUDNN_PARAM_DATA_UINT64;
  offset_val_ = offset;
  offset_ptr_ = nullptr;
  sub_offset_ = 0;
  return MUDNN_STATUS_SUCCESS;
}

mudnnStatus_t Dropout::SetOffset(uint64_t* offset_ptr, uint64_t sub_offset) {
  offset_type_ = MUDNN_PARAM_DATA_POINTER;
  offset_ptr_ = offset_ptr;
  sub_offset_ = sub_offset;
  return MUDNN_STATUS_SUCCESS;
}

void Dropout::BuildForwardDescriptor() {
  void* seed = (seed_type_ == MUDNN_PARAM_DATA_POINTER)
      ? static_cast<void*>(seed_ptr_)
      : static_cast<void*>(&seed_val_);
  void* offset = (offset_type_ == MUDNN_PARAM_DATA_POINTER)
      ? static_cast<void*>(offset_ptr_)
      : static_cast<void*>(&offset_val_);

  double fwd_scale = (p_ < 1.0) ? (1.0 / (1.0 - p_)) : 0.0;

  CHECK_MUDNN_STATUS(
      mudnnSetDropoutDescriptorEx(
          Desc(),
          p_,
          fwd_scale,
          seed_type_,
          seed,
          offset_type_,
          offset,
          sub_offset_),
      "mudnnSetDropoutDescriptorEx");
}

void Dropout::BuildBackwardDescriptor() {
  uint64_t dummy = 0;
  CHECK_MUDNN_STATUS(
      mudnnSetDropoutDescriptorEx(
          Desc(),
          0.0,
          scale_,
          MUDNN_PARAM_DATA_UINT64,
          &dummy,
          MUDNN_PARAM_DATA_UINT64,
          &dummy,
          0),
      "mudnnSetDropoutDescriptorEx");
}

mudnnStatus_t Dropout::RunDropout(
    muHandle& h,
    muTensor& out,
    const muTensor& in,
    muTensor& mask) {
  BuildForwardDescriptor();

  out.Build();
  in.Build();
  mask.Build();

  return mudnnDropoutForwardEx(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(in),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(mask));
}

mudnnStatus_t Dropout::RunDropoutBwd(
    muHandle& h,
    muTensor& out,
    const muTensor& grad,
    const muTensor& mask) {
  BuildBackwardDescriptor();

  out.Build();
  grad.Build();
  mask.Build();

  return mudnnDropoutBackwardEx(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(grad),
      MUDNN_UNPACK_TENSOR(mask));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // namespace at::musa
