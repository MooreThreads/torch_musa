#include "torch_musa/csrc/aten/mudnn/Reduce.h"

#include <optional>

#include "torch_musa/csrc/aten/mudnn/Exception.h"
#include "torch_musa/csrc/aten/mudnn/Param.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnReduceMode_t GetReduceModeFromEnum(Reduce::Mode mode) {
#define CASE(MODE1, MODE2)    \
  case Reduce::Mode::MODE1: { \
    return MODE2;             \
  }

  switch (mode) {
    MUDNN_FORALL_REDUCE_OPS(CASE)
    default:
      throw std::runtime_error("unreachable");
  }
#undef CASE
}

void SetReduceImpl(
    Reduce& op,
    Reduce::Mode mode,
    int ndim,
    const int* dim,
    std::optional<float> ord,
    std::optional<int> correction) {
  const auto m = GetReduceModeFromEnum(mode);
  const auto o = ord.value_or(2.0f);
  const auto c = correction.value_or(1);
  CHECK_MUDNN_STATUS(
      mudnnSetReduceDescriptor(op.Desc(), m, ndim, dim, o, c),
      "mudnnSetReduceDescriptor");
}

#else

void SetReduceImpl(
    Reduce& op,
    Reduce::Mode mode,
    int ndim,
    const int* dim,
    std::optional<float> ord,
    std::optional<int> correction) {
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  CHECK_MUDNN_STATUS(op.SetDim(ndim, dim), "SetDim");
  if (correction) {
    CHECK_MUDNN_STATUS(op.SetCorrection(correction.value()), "SetCorrection");
  }
  if (ord) {
    CHECK_MUDNN_STATUS(op.SetNormOrd(ord.value()), "SetNormOrd");
  }
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t Reduce::Run(
    muHandle& h,
    muTensor& out,
    const muTensor& in,
    const MemoryMaintainer& maintainer) const {
  out.Build();
  in.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetReduceWorkspaceSize(h, Desc(), &ws_size, out.Desc(), in.Desc()),
      "mudnnGetReduceWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnReduce(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(in),
      mem.get(),
      ws_size,
      MUDNN_UNPACK_TENSOR(out));
}

mudnnStatus_t Reduce::RunIndices(
    muHandle& h,
    muTensor& out,
    const muTensor& in,
    const MemoryMaintainer& maintainer) const {
  out.Build();
  in.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetReduceIndicesWorkspaceSize(
          h, Desc(), ws_size, out.Desc(), in.Desc()),
      "mudnnGetReduceIndicesWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnReduceOnlyIndices(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(in),
      mem.get(),
      ws_size,
      MUDNN_UNPACK_TENSOR(out));
}

mudnnStatus_t Reduce::RunWithIndices(
    muHandle& h,
    muTensor& out,
    muTensor& indices,
    const muTensor& in,
    const MemoryMaintainer& maintainer) const {
  out.Build();
  indices.Build();
  in.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetReduceWithIndicesWorkspaceSize(
          h, Desc(), ws_size, out.Desc(), indices.Desc(), in.Desc()),
      "mudnnGetReduceWithIndicesWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnReduceWithIndices(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(in),
      mem.get(),
      ws_size,
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(indices));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetReduce(Reduce& op, Reduce::Mode mode, int ndim, const int* dim) {
  SetReduceImpl(op, mode, ndim, dim, std::nullopt, std::nullopt);
}

void SetReduceCorrection(
    Reduce& op,
    Reduce::Mode mode,
    int ndim,
    const int* dim,
    int correction) {
  SetReduceImpl(op, mode, ndim, dim, std::nullopt, correction);
}

void SetReduceNorm(
    Reduce& op,
    Reduce::Mode mode,
    int ndim,
    const int* dim,
    float ord) {
  SetReduceImpl(op, mode, ndim, dim, ord, std::nullopt);
}

} // namespace at::musa
