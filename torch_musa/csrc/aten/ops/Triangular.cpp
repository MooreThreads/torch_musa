#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/tril_native.h>
#include <ATen/ops/triu_native.h>
#endif
#include <mudnn.h>
#include <string>

namespace at {
namespace musa {
namespace {
struct structured_triu_musa_out final
    : public at::native::structured_triu_cuda {
  structured_triu_musa_out(Tensor& out0) : outputs_{std::ref(out0)} {}
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
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

at::Tensor TriuPorting(const Tensor& self, int64_t diagonal, Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, out, "MUSA_triu_out_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "MUSA_triu_out_out", "self");
  structured_triu_musa_out op(out);
  op.meta(self, diagonal);
  op.impl(self, diagonal, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}

struct structured_tril_musa_out final
    : public at::native::structured_tril_cuda {
  structured_tril_musa_out(Tensor& out0) : outputs_{std::ref(out0)} {}
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
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

Tensor TrilPorting(const Tensor& self, int64_t diagonal, Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, out, "MUSA_tril_out_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "MUSA_tril_out_out", "self");
  structured_tril_musa_out op(out);
  op.meta(self, diagonal);
  op.impl(self, diagonal, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}

} // namespace

void TriCallOut(
    Tensor& out,
    const Tensor& input,
    ::musa::dnn::TriangularMat::Mode mode,
    const int64_t diag,
    const std::string name) {
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of " + name + " must be MUSA, but now is ",
      input.device());
  // TODO(@caizhi): The original strategy was to execute porting when mudnn
  // failed. However mudnn will have an error log when failing. When mudnn sets
  // the error log level, this strategy will be turned on.
  if (mode == ::musa::dnn::TriangularMat::Mode::TRIU) {
    TriuPorting(input, diag, out);
  } else {
    TrilPorting(input, diag, out);
  }
#if 0
  Tensor contiguous_input = input.contiguous();
  muHandle& h = GetMudnnHandle();
  auto mudnn_out = CreateMUTensor(out);
  auto mudnn_input = CreateMUTensor(contiguous_input);
  ::musa::dnn::TriangularMat op;
  op.SetMode(mode);
  op.SetDiagonal(diag);
  if (op.Run(h, mudnn_out, mudnn_input) != ::musa::dnn::Status::SUCCESS) {
    if (mode == ::musa::dnn::TriangularMat::Mode::TRIU) {
      TriuPorting(input, diag, out);
    } else {
      TrilPorting(input, diag, out);
    }
  }
#endif
}

Tensor Triu(const Tensor& self, int64_t diagonal = 0) {
  c10::musa::MUSAGuard device_guard(self.device());
  Tensor output = at::empty_like(
      self, self.options().memory_format(at::MemoryFormat::Contiguous));
  TriCallOut(
      output, self, ::musa::dnn::TriangularMat::Mode::TRIU, diagonal, "Triu");
  return output;
}

Tensor& Triu_(Tensor& self, int64_t diagonal = 0) {
  c10::musa::MUSAGuard device_guard(self.device());
  TriCallOut(
      self, self, ::musa::dnn::TriangularMat::Mode::TRIU, diagonal, "Triu");
  return self;
}

Tensor& TriuOut(const Tensor& self, int64_t diagonal, Tensor& output) {
  c10::musa::MUSAGuard device_guard(self.device());
  output.resize_(self.sizes());
  TriCallOut(
      output, self, ::musa::dnn::TriangularMat::Mode::TRIU, diagonal, "Triu");
  return output;
}

Tensor Tril(const Tensor& self, int64_t diagonal = 0) {
  c10::musa::MUSAGuard device_guard(self.device());
  Tensor output = at::empty_like(
      self, self.options().memory_format(at::MemoryFormat::Contiguous));
  TriCallOut(
      output, self, ::musa::dnn::TriangularMat::Mode::TRIL, diagonal, "Tril");
  return output;
}

Tensor& Tril_(Tensor& self, int64_t diagonal = 0) {
  c10::musa::MUSAGuard device_guard(self.device());
  TriCallOut(
      self, self, ::musa::dnn::TriangularMat::Mode::TRIL, diagonal, "Tril");
  return self;
}

Tensor& TrilOut(const Tensor& self, int64_t diagonal, Tensor& output) {
  c10::musa::MUSAGuard device_guard(self.device());
  output.resize_(self.sizes());
  TriCallOut(
      output, self, ::musa::dnn::TriangularMat::Mode::TRIL, diagonal, "Tril");
  return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("triu", &Triu);
  m.impl("triu_", &Triu_);
  m.impl("triu.out", &TriuOut);

  m.impl("tril_", Tril_);
  m.impl("tril.out", TrilOut);
  m.impl("tril", Tril);
}

} // namespace musa
} // namespace at
