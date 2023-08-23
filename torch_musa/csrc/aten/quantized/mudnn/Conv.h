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

enum class ActMode { IDENTITY, RELU, SILU };

template <int kSpatialDim>
at::SmallVector<int64_t, kSpatialDim + 2> MakeQConvOutputShape(
    int N, // mini-batch
    int M, // output channels
    const std::array<int64_t, kSpatialDim>& input_image_shape,
    const std::vector<int64_t>& kernel,
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& padding,
    const torch::List<int64_t>& dilation);

void inline SetMudnnQuantizationInfo(
    at::musa::muTensor& self,
    double scales,
    int64_t zero_points) {
  float scales_ = static_cast<float>(scales);
  unsigned int zero_points_ = static_cast<unsigned int>(zero_points);
  CHECK_MUDNN_STATUS(
      self.SetQuantizationInfo(1, &scales_, &zero_points_),
      "Set quantization info");
}

void inline ConfigConv(
    ::musa::dnn::Convolution& c,
    torch::List<int64_t> padding,
    torch::List<int64_t> stride,
    torch::List<int64_t> dilation,
    int64_t groups) {
  CHECK_MUDNN_STATUS(c.SetGroups(groups), "SetGroups");
  CHECK_MUDNN_STATUS(
      c.SetComputeMode(::musa::dnn::Convolution::ComputeMode::TENSOR),
      "SetComputeMode");

  int sizes = padding.size();
  if (sizes == 2) {
    std::vector<int> pad_ = {
        static_cast<int>(padding[0]), static_cast<int>(padding[1])};
    std::vector<int> str_ = {
        static_cast<int>(stride[0]), static_cast<int>(stride[1])};
    std::vector<int> dil_ = {
        static_cast<int>(dilation[0]), static_cast<int>(dilation[1])};
    CHECK_MUDNN_STATUS(
        c.SetNdInfo(sizes, pad_.data(), str_.data(), dil_.data()), "SetNdInfo");
  } else {
    TORCH_CHECK(false, "We currently only support quantized conv2d");
  }
}

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
      : weight_(std::move(orig_weight)),
        bias_(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        output_padding_(std::move(output_padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        transpose_(transpose),
        q_scheme_(q_scheme),
        output_channels_(output_channels) {
  } // output channel needs to be stored when we have to pad this dimension

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_silu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);

  at::Tensor apply_add(
      const at::Tensor& input,
      const at::Tensor& accum,
      double output_scale,
      int64_t output_zero_point);

  at::Tensor apply_add_relu(
      const at::Tensor& input,
      const at::Tensor& accum,
      double output_scale,
      int64_t output_zero_point);

  at::Tensor apply_add_silu(
      const at::Tensor& input,
      const at::Tensor& accum,
      double output_scale,
      int64_t output_zero_point);

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
  at::Tensor weight_;
  c10::optional<at::Tensor> bias_;
  torch::List<int64_t> stride_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> output_padding_;
  torch::List<int64_t> dilation_;
  int64_t groups_;
  bool transpose_;
  c10::QScheme q_scheme_;
  int64_t output_channels_;

  template <ActMode act_mode>
  at::Tensor apply_impl(
      const at::Tensor& input,
      const c10::optional<at::Tensor>& accum,
      double output_scale,
      int64_t output_zero_point);

  template <ActMode act_mode>
  void apply_impl_helper(
      at::Tensor& quantized_output,
      const at::Tensor& input,
      const c10::optional<at::Tensor>& accum,
      double output_scale,
      int64_t output_zero_point);
};
