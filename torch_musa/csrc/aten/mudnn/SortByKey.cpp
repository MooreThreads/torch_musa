#include "torch_musa/csrc/aten/mudnn/SortByKey.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

void SetSortByKeyImpl(SortByKey& op, int dim, bool stable, bool descending) {
  CHECK_MUDNN_STATUS(
      mudnnSetSortByKeyDescriptor(op.Desc(), dim, stable, descending),
      "mudnnSetSortByKeyDescriptor");
}

#else

void SetSortByKeyImpl(SortByKey& op, int dim, bool stable, bool descending) {
  CHECK_MUDNN_STATUS(op.SetDim(dim), "SetDim");
  CHECK_MUDNN_STATUS(op.SetStable(stable), "SetStable");
  CHECK_MUDNN_STATUS(op.SetDescending(descending), "SetDescending");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t SortByKey::Run(
    muHandle& h,
    muTensor& keys_out,
    muTensor& values_out,
    const muTensor& keys_in,
    const muTensor& values_in,
    const MemoryMaintainer& maintainer) const {
  keys_out.Build();
  values_out.Build();
  keys_in.Build();
  values_in.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetSortByKeyWorkspaceSize(
          h,
          Desc(),
          keys_out.Desc(),
          values_out.Desc(),
          keys_in.Desc(),
          values_in.Desc(),
          &ws_size),
      "mudnnGetSortByKeyWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnSortByKey(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(keys_in),
      MUDNN_UNPACK_TENSOR(values_in),
      mem.get(),
      ws_size,
      MUDNN_UNPACK_TENSOR(keys_out),
      MUDNN_UNPACK_TENSOR(values_out));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetSortByKey(SortByKey& op, int dim, bool stable, bool descending) {
  SetSortByKeyImpl(op, dim, stable, descending);
}

} // namespace at::musa
