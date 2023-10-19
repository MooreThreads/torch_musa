#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Pool.h>
#include <ATen/native/ScatterGatherChecks.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

#include <mudnn.h>

namespace at {
namespace musa {

namespace {
struct structured_gather_out_out final
    : public at::native::structured_gather_out {
  structured_gather_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
    if (C10_UNLIKELY(maybe_proxy.has_value())) {
      proxy_outputs_[output_idx] =
          c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());
    }
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
  }
  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};
at::Tensor& wrapper_MUSA_gather_out_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_MUSA_gather_out_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_MUSA_gather_out_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "wrapper_MUSA_gather_out_out", "index");
  structured_gather_out_out op(out);
  op.meta(self, dim, index, sparse_grad);
  op.impl(self, dim, index, sparse_grad, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}
} // namespace

at::Tensor& GatherOut(
    const at::Tensor& input,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& out) {
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of Gather must be MUSA, but now is ",
      input.device());
  TORCH_CHECK(
      index.device().type() == kMUSA,
      "Device of index tensor of Gather must be MUSA, but now is ",
      index.device());
  TORCH_CHECK(
      out.device().type() == kMUSA,
      "Device of out tensor of Gather must be MUSA, but now is ",
      out.device());
  TORCH_CHECK(
      !sparse_grad,
      "That parameter sparse_grad of Gather is True is not supported now!");

  if (input.scalar_type() == at::kHalf) {
    return wrapper_MUSA_gather_out_out(input, dim, index, sparse_grad, out);
  }
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float ||
          input.scalar_type() == at::ScalarType::Long,
      "Dtype of input tensor of Gather only support Float32 and Int64, "
      "but now it is ",
      input.scalar_type());
  c10::musa::MUSAGuard device_guard(input.device());
  auto contiguous_input = input.contiguous();
  auto contiguous_index = index.contiguous();

  out.resize_(contiguous_index.sizes());

  if (contiguous_index.numel() != 0) {
    TORCH_CHECK(
        contiguous_index.scalar_type() == at::ScalarType::Long,
        "gather",
        "(): Expected dtype int64 for index");
    const int64_t wrapped_dim = at::maybe_wrap_dim(dim, contiguous_input.dim());
    at::native::gather_shape_check(
        contiguous_input, wrapped_dim, contiguous_index);
  } else {
    return out;
  }

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::GatherX gather_op;
  auto mt_input = CreateMUTensor(contiguous_input);
  auto mt_index = CreateMUTensor(contiguous_index);
  auto mt_out = CreateMUTensor(out);
  CHECK_MUDNN_STATUS(gather_op.SetAxis(dim), "SetAxis");
  CHECK_MUDNN_STATUS(
      gather_op.SetMode(::musa::dnn::GatherX::Mode::GATHER_ELEMENTS),
      "SetMode");
  CHECK_MUDNN_STATUS(gather_op.Run(h, mt_out, mt_index, mt_input), "Run");
  return out;
}

at::Tensor Gather(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad) {
  c10::musa::MUSAGuard device_guard(self.device());
  Tensor result = at::empty(index.sizes(), self.options());
  GatherOut(self, dim, index, sparse_grad, result);
  return result;
}

ADVANCED_REGISTER(aten, PrivateUse1, "gather.out", GatherOut)
ADVANCED_REGISTER(aten, PrivateUse1, "gather", Gather)

} // namespace musa
} // namespace at
