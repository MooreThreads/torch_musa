#include <torch_musa/csrc/aten/mudnn/Concat.h>

#include <vector>

#include <torch_musa/csrc/aten/mudnn/Exception.h>

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

void SetConcatImpl(Concat& op, int axis) {
  CHECK_MUDNN_STATUS(
      mudnnSetConcatDescriptor(op.Desc(), axis), "mudnnSetConcatDescriptor");
}

#else

void SetConcatImpl(Concat& op, int axis) {
  CHECK_MUDNN_STATUS(op.SetAxis(axis), "SetAxis");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t Concat::Run(
    muHandle& h,
    muTensor& out,
    int num_input,
    const muTensor* ins) const {
  out.Build();

  std::vector<mudnnTensorDescriptor_t> inputDescs(num_input);
  std::vector<const void*> inputPtrs(num_input);
  for (int i = 0; i < num_input; ++i) {
    ins[i].Build();
    inputDescs[i] = ins[i].Desc();
    inputPtrs[i] = ins[i].DataPtr();
  }

  return mudnnConcat(
      h,
      Desc(),
      inputDescs.data(),
      inputPtrs.data(),
      num_input,
      out.Desc(),
      out.DataPtr());
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetConcat(Concat& op, int axis) {
  SetConcatImpl(op, axis);
}

} // namespace at::musa
