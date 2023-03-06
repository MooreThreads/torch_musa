#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_TRIANGULAR_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_TRIANGULAR_H_

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

DECLARE_DISPATCH(void (*)(Tensor&, const Tensor&, const int64_t), triu_stub);
DECLARE_DISPATCH(void (*)(Tensor&, const Tensor&, const int64_t), tril_stub);

enum class TriangularMode { TRIU, TRIL };

} // namespace native
} // namespace at

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_TRIANGULAR_H_
