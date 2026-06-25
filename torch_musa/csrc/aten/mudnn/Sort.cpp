#include "torch_musa/csrc/aten/mudnn/Sort.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t MudnnSort::Run(
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
      mudnnGetSortWorkspaceSize(
          h, Desc(), values.Desc(), indices.Desc(), input.Desc(), &ws_size),
      "mudnnGetSortWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnSort(
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
void SetSortImpl(
    ::musa::dnn::Sort& op,
    int64_t dim,
    bool descending,
    bool stable) {
  CHECK_MUDNN_STATUS(
      mudnnSetSortDescriptor(
          op.Desc(), static_cast<int>(dim), stable, descending),
      "mudnnSetSortDescriptor");
}
#else
void SetSortImpl(
    ::musa::dnn::Sort& op,
    int64_t dim,
    bool descending,
    bool stable) {
  CHECK_MUDNN_STATUS(op.SetDim(static_cast<int>(dim)), "SetDim");
  CHECK_MUDNN_STATUS(op.SetDescending(descending), "SetDescending");
  CHECK_MUDNN_STATUS(op.SetStable(stable), "SetStable");
}
#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

void SetSort(::musa::dnn::Sort& op, int64_t dim, bool descending, bool stable) {
  return SetSortImpl(op, dim, descending, stable);
}

} // namespace at::musa
