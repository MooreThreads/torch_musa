#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_REDUCE_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_REDUCE_H_

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native { // this namespace is used to declare logsumexp stub only

DECLARE_DISPATCH(void (*)(Tensor&, const Tensor&, int64_t), logsumexp_stub);
}
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_REDUCE_H_
