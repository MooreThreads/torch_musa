#include "torch_musa/csrc/aten/mudnn/Fill.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"
#include "torch_musa/csrc/aten/mudnn/Param.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

template <typename T>
void SetFillImpl(::musa::dnn::Fill& op, T value) {
  MudnnParam v;
  v.Set(value);
  CHECK_MUDNN_STATUS(
      mudnnSetFillDescriptor(op.Desc(), v.Type(), v.Ptr()),
      "mudnnSetFillDescriptor");
}

#else

template <typename T>
void SetFillImpl(::musa::dnn::Fill& op, T value) {
  CHECK_MUDNN_STATUS(op.SetValue(value), "SetValue");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t MudnnFill::Run(muHandle& h, muTensor& out) const {
  out.Build();
  return mudnnFill(h, Desc(), nullptr, nullptr, MUDNN_UNPACK_TENSOR(out));
}

mudnnStatus_t MudnnFill::Run(muHandle& h, muTensor& out, muTensor& mask) const {
  out.Build();
  mask.Build();
  return mudnnFill(
      h, Desc(), MUDNN_UNPACK_TENSOR(mask), MUDNN_UNPACK_TENSOR(out));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetFill(::musa::dnn::Fill& op, int64_t value) {
  SetFillImpl<int64_t>(op, value);
}

void SetFill(::musa::dnn::Fill& op, double value) {
  SetFillImpl<double>(op, value);
}

} // namespace at::musa
