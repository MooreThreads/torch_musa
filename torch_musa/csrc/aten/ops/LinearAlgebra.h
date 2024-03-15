#ifndef ATEN_SRC_ATEN_NATIVE_MUSA_LINEARALGEBRA_H_
#define ATEN_SRC_ATEN_NATIVE_MUSA_LINEARALGEBRA_H_

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

DECLARE_DISPATCH(void (*)(Tensor&, const Tensor&), inverse_stub);

} // namespace at::native

#endif // ATEN_SRC_ATEN_NATIVE_MUSA_LINEARALGEBRA_H_
