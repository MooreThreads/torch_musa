#include <ATen/ATen.h>
#include <ATen/native/quantized/PackedParams.h>
#include <torch/library.h>

#include <tuple>

#include "torch_musa/csrc/aten/quantized/mudnn/Linear.h"

std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedLinearWeightMudnn::
    unpack() {
  return std::tuple<at::Tensor, c10::optional<at::Tensor>>{orig_weight, bias_};
}
