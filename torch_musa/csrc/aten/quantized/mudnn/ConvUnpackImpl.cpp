#include <ATen/ATen.h>
#include <ATen/native/quantized/PackedParams.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/quantized/mudnn/Conv.h"

#include <tuple>

template <int kSpatialDim>
std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeightMudnn<
    kSpatialDim>::unpack() {
  return std::tuple<at::Tensor, c10::optional<at::Tensor>>{weight_, bias_};
}

template std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeightMudnn<
    2>::unpack();
