
#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_BUCKETIZE_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_BUCKETIZE_H_

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace musa {

// for clamp op to call
Tensor& MaximumTensorOut(
    const Tensor& self,
    const Tensor& other,
    Tensor& output);
Tensor& MinimumTensorOut(
    const Tensor& self,
    const Tensor& other,
    Tensor& output);

} // namespace musa
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_BUCKETIZE_H_
