#include <ATen/Config.h>
#include <ATen/native/ReduceOpsUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cumprod_native.h>
#include <ATen/ops/cumsum_native.h>
#endif

#include "torch_musa/csrc/aten/mudnn/Cum.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"
#include "torch_musa/csrc/aten/utils/MudnnUtils.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at::musa {

namespace {

using CUM_MODE = ::musa::dnn::Cum::Mode;

void CumulativeImpl(Tensor& out, const Tensor& in, int64_t dim, CUM_MODE mode) {
  namedinference::propagate_names(out, in);

  if (in.numel() == 0) {
    return;
  }
  if (in.dim() == 0) {
    out.fill_(in);
  }

  const c10::musa::MUSAGuard device_guard(in.device());
  auto in_ctype = in.scalar_type();
  auto out_ctype = out.scalar_type();

  if (isFloatingType(in_ctype) || isFloatingType(out_ctype)) {
    out_ctype = promoteTypes(in_ctype, out_ctype);
    in_ctype = out_ctype;
  } else if (out_ctype == ScalarType::Int || out_ctype == ScalarType::Long) {
    if (in_ctype == ScalarType::Byte || in_ctype == ScalarType::Short) {
      in_ctype = ScalarType::Int;
    } else if (in_ctype == ScalarType::Long) {
      out_ctype = in_ctype;
    }
  } else {
    // Special cases for muDNN workarounds.
    if (in_ctype == ScalarType::Long) {
      out_ctype = in_ctype;
    } else {
      out_ctype = ScalarType::Int;
      if (in_ctype == ScalarType::Byte || in_ctype == ScalarType::Short) {
        in_ctype = out_ctype;
      }
    }
  }

  using Proxy = typename c10::MaybeOwned<Tensor>;
  auto make_proxy = [](const Tensor& in, ScalarType dtype) -> Proxy {
    if (dtype != in.scalar_type()) {
      return Proxy::owned(in.to(dtype));
    }
    return Proxy::borrowed(in);
  };
  auto c_in = make_proxy(in, in_ctype);
  auto c_out = make_proxy(out, out_ctype);
  auto contig_c_in = FormatContiguous(*c_in, MemoryFormat::Contiguous);
  auto contig_c_out = FormatContiguous(*c_out, MemoryFormat::Contiguous);

  auto mudnn_out = CreateMUTensor(contig_c_out);
  auto mudnn_in = CreateMUTensor(contig_c_in);
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Cum cum;
  SetCum(cum, static_cast<int64_t>(dim), mode);
  CHECK_MUDNN_STATUS(
      cum.Run(h, mudnn_out, mudnn_in, InternalMemAlloc), "CumRun");

  if (out_ctype != out.scalar_type()) {
    out.copy_(contig_c_out);
  }
}

Tensor& CumulativeOut(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype,
    Tensor& out,
    CUM_MODE mode) {
  // Keep logical consistency with torch.
  if (dtype) {
    const auto out_dtype = out.scalar_type();
    const auto expect_dtype = dtype.value();
    TORCH_CHECK(
        expect_dtype == out_dtype,
        "Expected out tensor to have dtype ",
        expect_dtype,
        ", but got ",
        out_dtype,
        " instead");
  }
  native::resize_output(out, self.sizes());
  CumulativeImpl(out, self, dim, mode);
  return out;
}

Tensor& Cumulative_(
    Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype,
    CUM_MODE mode) {
  // Since check_inplace(...) does not modify the `self` tensor in torch,
  // we ignore the `dtype` parameter here, even though it may have a value.
  CumulativeImpl(self, self, dim, mode);
  return self;
}

Tensor Cumulative(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype,
    CUM_MODE mode) {
  const auto in_dtype = self.scalar_type();
  const auto out_dtype = dtype.value_or(
      isIntegralType(in_dtype, true) ? ScalarType::Long : in_dtype);
  Tensor out = at::empty(self.sizes(), self.options().dtype(out_dtype));
  CumulativeImpl(out, self, dim, mode);
  return out;
}

} // anonymous namespace

#define GEN_CUMULATIVE_FUNCTION(OP, MODE)                                     \
  Tensor OP(                                                                  \
      const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {     \
    return Cumulative(self, dim, dtype, MODE);                                \
  }                                                                           \
                                                                              \
  Tensor& OP##_(Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) { \
    return Cumulative_(self, dim, dtype, MODE);                               \
  }                                                                           \
                                                                              \
  Tensor& OP##Out(                                                            \
      const Tensor& self,                                                     \
      int64_t dim,                                                            \
      c10::optional<ScalarType> dtype,                                        \
      Tensor& out) {                                                          \
    return CumulativeOut(self, dim, dtype, out, MODE);                        \
  }

GEN_CUMULATIVE_FUNCTION(CumSum, CUM_MODE::ADD)
GEN_CUMULATIVE_FUNCTION(CumProd, CUM_MODE::MUL)

#undef GEN_CUMULATIVE_FUNCTION

} // namespace at::musa
