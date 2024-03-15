#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_RANGEFACTORIES_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_RANGEFACTORIES_H_

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

DECLARE_DISPATCH(
    void (*)(const Scalar&, const Scalar&, const Scalar&, Tensor&),
    arange_start_out_stub);
} // namespace native
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_RANGEFACTORIES_H_
