#include "torch_musa/csrc/aten/mudnn/TopK.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t TopK::Run(
    muHandle& h,
    muTensor& values,
    muTensor& indices,
    const muTensor& input,
    const MemoryMaintainer& maintainer) const {
  values.Build();
  indices.Build();
  input.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetTopKWorkspaceSize(
          h, Desc(), input.Desc(), values.Desc(), indices.Desc(), &ws_size),
      "mudnnGetTopKWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnTopK(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(input),
      mem.get(),
      ws_size,
      MUDNN_UNPACK_TENSOR(values),
      MUDNN_UNPACK_TENSOR(indices));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API
void SetTopKImpl(TopK& op, int64_t k, int64_t dim, bool largest, bool sorted) {
  CHECK_MUDNN_STATUS(
      mudnnSetTopKDescriptor(
          op.Desc(),
          static_cast<int>(k),
          static_cast<int>(dim),
          largest,
          sorted),
      "mudnnSetTopKDescriptor");
}
#else
void SetTopKImpl(TopK& op, int64_t k, int64_t dim, bool largest, bool sorted) {
  CHECK_MUDNN_STATUS(op.SetK(static_cast<int>(k)), "SetK");
  CHECK_MUDNN_STATUS(op.SetDim(static_cast<int>(dim)), "SetDim");
  CHECK_MUDNN_STATUS(op.SetLargest(largest), "SetLargest");
  CHECK_MUDNN_STATUS(op.SetSorted(sorted), "SetSorted");
}
#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

void SetTopK(TopK& op, int64_t k, int64_t dim, bool largest, bool sorted) {
  return SetTopKImpl(op, k, dim, largest, sorted);
}

} // namespace at::musa
