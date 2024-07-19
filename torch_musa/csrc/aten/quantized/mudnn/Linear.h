#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include "torch_musa/csrc/aten/ops/TensorFactory.h"

struct PackedLinearWeightMudnn : public LinearPackedParamsBase {
  PackedLinearWeightMudnn(
      at::Tensor orig_weight,
      c10::optional<at::Tensor> bias,
      c10::QScheme q_scheme)
      : orig_weight(std::move(orig_weight)),
        bias_(std::move(bias)),
        q_scheme(std::move(q_scheme)) {}

  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;
  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range = false)
      override {
    throw std::runtime_error(
        "apply_relu_out is not implemented for this packed "
        "parameter type");
  }
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range = false)
      override {
    throw std::runtime_error(
        "apply_relu_out is not implemented for this packed "
        "parameter type");
  }

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override;

  c10::optional<at::Tensor> bias() override {
    return bias_;
  }

  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias);

 private:
  at::Tensor orig_weight;
  c10::optional<at::Tensor> bias_;
  c10::QScheme q_scheme;

  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);

  template <bool ReluFused>
  void apply_impl_helper(
      at::Tensor& quantized_output,
      const at::Tensor& input,
      double output_scale,
      int64_t zero_point);
};
