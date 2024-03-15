#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_ONEHOT_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_ONEHOT_H_

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

DECLARE_DISPATCH(void (*)(Tensor&, const Tensor&, int), onehot_stub);

} // namespace native
namespace musa {

Tensor OneHot(const Tensor&, int64_t num_classes = -1);

} // namespace musa
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_ONEHOT_H_
