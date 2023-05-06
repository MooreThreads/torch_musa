#include <ATen/native/Resize.h>
#include <ATen/ops/cumsum_native.h>
#include <c10/util/Exception.h>

namespace at {
namespace native {
namespace musa {

extern Tensor create_out(
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options);

namespace {
void resize_out(
    const Tensor& out,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options) {
  TORCH_CHECK(
      options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ",
      options.dtype(),
      ", but got ",
      out.dtype(),
      " instead");
  TORCH_CHECK(
      options.device() == out.device(),
      "Expected out tensor to have device ",
      options.device(),
      ", but got ",
      out.device(),
      " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      // TODO: avoid the redispatch here
      out.as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(
          *options.memory_format_opt());
    }
  }
}
c10::optional<Tensor> maybe_create_proxy(
    const Tensor& out,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options) {
  if (out.strides() != strides) {
    return create_out(sizes, strides, options);
  }
  return c10::nullopt;
}
} // namespace

Tensor Tril(const Tensor& self, int64_t diagonal) {
  // FIXME(caizhi): here is a bug which is being fixed.
  (void)self;
  (void)diagonal;
  C10_THROW_ERROR(NotImplementedError, "Tril Op is not supported for now!");
  // structured_Tril_out op(out);
  // op.meta(self, dim, dtype);
  // op.impl(self, dim, dtype, op.maybe_get_output(0));
  // if (op.proxy_outputs_[0].has_value())
  //   op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return self;
}
} // namespace musa
} // namespace native
} // namespace at
