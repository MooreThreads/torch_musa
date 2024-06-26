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

using embedding_dense_backward_fn =
    Tensor (*)(const Tensor&, const Tensor&, int64_t, int64_t, bool);

DECLARE_DISPATCH(embedding_dense_backward_fn, embedding_dense_backward_stub);

} // namespace native
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_EMBEDDING_H_
