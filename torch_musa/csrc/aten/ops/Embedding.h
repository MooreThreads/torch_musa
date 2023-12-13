#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_EMBEDDING_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_EMBEDDING_H_

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

DECLARE_DISPATCH(
    void (*)(
        Tensor&,
        const Tensor&,
        const Tensor&,
        const Tensor&,
        const int64_t,
        const int64_t),
    embedding_bag_stub);
} // namespace native
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_EMBEDDING_H_
