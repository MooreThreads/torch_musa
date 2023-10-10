#include <ATen/ATen.h>
#include <ATen/native/quantized/PackedParams.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/quantized/mudnn/Conv.h"

#include <tuple>

template <int kSpatialDim>
std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeightMudnn<
    kSpatialDim>::unpack() {
  // weight is in NHWC format and shape to speed up inference, if we need to
  // fetch weight from outside, we should permute it to NCHW shape to keep
  // consistence with pytorch
  at::Tensor rt_weight = weight_.permute({0, 3, 1, 2});
  return std::tuple<at::Tensor, c10::optional<at::Tensor>>{rt_weight, bias_};
}

template std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeightMudnn<
    2>::unpack();
