#include <ATen/TensorNames.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorCompare.h>

namespace at {
namespace native {

void IsinMUSAKernel(
    const Tensor& elements,
    const Tensor& test_elements,
    bool invert,
    const Tensor& out) {
  std::vector<int64_t> bc_shape(elements.dim(), 1);
  bc_shape.push_back(-1);
  out.copy_(
      invert ? elements.unsqueeze(-1).ne(test_elements.view(bc_shape)).all(-1)
             : elements.unsqueeze(-1).eq(test_elements.view(bc_shape)).any(-1));
}

REGISTER_MUSA_DISPATCH(isin_default_stub, &IsinMUSAKernel);

} // namespace native

namespace musa {

std::tuple<Tensor&, Tensor&> ModeOut(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  TORCH_CHECK(
      self.layout() == Layout::Strided,
      "mode only supports strided layout, got: ",
      self.layout());
  TORCH_CHECK(
      self.device() == values.device(),
      "expected device '",
      self.device(),
      "' but got '",
      values.device(),
      "' for values output");
  TORCH_CHECK(
      self.device() == indices.device(),
      "expected device '",
      self.device(),
      "' but got '",
      indices.device(),
      "' for indices output");
  TORCH_CHECK(
      self.scalar_type() == values.scalar_type(),
      "expected scalar type '",
      self.scalar_type(),
      "' but got '",
      values.scalar_type(),
      "' for values output");
  TORCH_CHECK(
      indices.scalar_type() == ScalarType::Long,
      "expected scalar type '",
      ScalarType::Long,
      "' but got '",
      indices.scalar_type(),
      "' for indices output");
  dim = maybe_wrap_dim(dim, self.dim());
  if (self.numel() == 0) {
    auto sizes =
        at::native::get_zero_numel_tensor_size(self, dim, keepdim, "mode()");
    at::native::resize_output(values, sizes);
    at::native::resize_output(indices, sizes);
    return std::tie(values, indices);
  } else if (at::native::_dimreduce_return_trivial_no_ident(
                 values, self, dim, keepdim, "mode")) {
    AT_ASSERT(values.dim() == 0);
    indices.resize_({}).fill_(0);
    return std::forward_as_tuple(values, indices);
  } else {
    auto result = [&]() {
      at::NoNamesGuard guard;
      at::native::mode_stub(
          self.device().type(), values, indices, self, dim, keepdim);
      return std::tuple<Tensor&, Tensor&>{values, indices};
    }();
    namedinference::propagate_names_for_reduction(
        std::get<0>(result), self, dim, keepdim);
    namedinference::propagate_names_for_reduction(
        std::get<1>(result), self, dim, keepdim);
    return result;
  }
}

std::tuple<Tensor, Tensor> Mode(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return ModeOut(self, dim, keepdim, values, indices);
}

} // namespace musa
} // namespace at
