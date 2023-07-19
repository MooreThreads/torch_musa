#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include "torch_musa/csrc/aten/ops/TensorFactory.h"

template <int kSpatialDim = 2>
struct TORCH_API PackedConvWeightMudnn
    : public ConvPackedParamsBase<kSpatialDim> {
  PackedConvWeightMudnn(
      at::Tensor orig_weight,
      c10::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose,
      c10::QScheme q_scheme,
      int64_t output_channels)
      : maybe_padded_weight_(std::move(orig_weight)),
        bias_(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        output_padding_(std::move(output_padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        transpose_(transpose),
        q_scheme_(q_scheme),
        num_unpadded_output_channels_(output_channels) {
  } // output channel needs to be stored when we have to pad this dimension

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_dynamic(const at::Tensor& input, bool reduce_range)
      override {
    TORCH_CHECK(false, "apply_dynamic is currently not reported");
  }

  at::Tensor apply_dynamic_relu(const at::Tensor& input, bool reduce_range) {
    TORCH_CHECK(false, "apply_dynamic_relu is currently not reported");
  }

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override;

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose);

  torch::List<int64_t> stride() const override {
    return stride_;
  }

  torch::List<int64_t> padding() const override {
    return padding_;
  }

  torch::List<int64_t> output_padding() const override {
    return output_padding_;
  }

  torch::List<int64_t> dilation() const override {
    return dilation_;
  }

  int64_t groups() const override {
    return groups_;
  }

  bool transpose() const override {
    return transpose_;
  }

 private:
  // cudnn needs conv2d's weight in **int8** dtype and output channel
  // to be a multiple of 4, and it would pad weight to be one if it is not
  // we currently follow what cudnn does and I (@fan.mo) didn't check that
  // of **mudnn**, so we keep the name *maybe*_padded_weight_ here.
  at::Tensor maybe_padded_weight_;
  c10::optional<at::Tensor> bias_;
  torch::List<int64_t> stride_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> output_padding_;
  torch::List<int64_t> dilation_;
  int64_t groups_;
  bool transpose_;
  c10::QScheme q_scheme_;
  int64_t num_unpadded_output_channels_;

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
      int64_t output_zero_point);
};
