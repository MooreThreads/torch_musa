#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_TENSORSHAPE_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_TENSORSHAPE_H_

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

// we could have used stubs defined IndexKernel.h but the function
// signature is kind different from what it was, so we define our
// stubs here
DECLARE_DISPATCH(
    void (*)(Tensor&, const std::vector<Tensor>&, const Tensor&, const bool),
    indexput_stub);
DECLARE_DISPATCH(
    void (*)(Tensor&, int, const std::vector<Tensor>&, const Tensor&),
    indexes_stub);
DECLARE_DISPATCH(
    void (*)(const int, Tensor&, const Tensor&, const Tensor&),
    indexselect_stub);
} // namespace native
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_TENSORSHAPE_H_
