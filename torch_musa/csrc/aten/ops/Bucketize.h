#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_BUCKETIZE_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_BUCKETIZE_H_

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

DECLARE_DISPATCH(
    void (*)(Tensor&, const Tensor&, const Tensor&, bool),
    bucketize_stub);

} // namespace native
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_BUCKETIZE_H_
