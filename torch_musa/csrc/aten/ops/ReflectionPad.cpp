#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/ops/reflection_pad1d_backward_native.h>
#include <mudnn.h>
#include <torch/library.h>
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {
using Pad = ::musa::dnn::Pad;
using Pad_MODE = ::musa::dnn::Pad::Mode;

void ConfigPad(Pad& op, IntArrayRef pad, Pad_MODE mode) {
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  switch (pad.size()) {
    case 2:
      CHECK_MUDNN_STATUS(
          op.SetPaddingInfo({
              static_cast<int>(pad[0]),
              static_cast<int>(pad[1]),
          }),
          "SetPaddingInfo");
      break;
    case 4:
      CHECK_MUDNN_STATUS(
          op.SetPaddingInfo(
              {static_cast<int>(pad[0]),
               static_cast<int>(pad[1]),
               static_cast<int>(pad[2]),
               static_cast<int>(pad[3])}),
          "SetPaddingInfo");
      break;
    case 6:
      CHECK_MUDNN_STATUS(
          op.SetPaddingInfo(
              {static_cast<int>(pad[0]),
               static_cast<int>(pad[1]),
               static_cast<int>(pad[2]),
               static_cast<int>(pad[3]),
               static_cast<int>(pad[4]),
               static_cast<int>(pad[5])}),
          "SetPaddingInfo");
      break;
    default:
      AT_ERROR("error dimension!");
      break;
  }
}

inline void CheckPad(const Tensor& self, IntArrayRef pad) {
  const auto input_dim = self.dim();
  TORCH_CHECK(pad.size() % 2 == 0, "Padding length must be divisible by 2");
  TORCH_CHECK(
      static_cast<int64_t>(pad.size()) <= input_dim * 2,
      "Padding length too large");
  TORCH_CHECK(
      (pad.size() == 2 && (input_dim == 2 || input_dim == 3)) ||
          (pad.size() == 4 && (input_dim == 3 || input_dim == 4)) ||
          (pad.size() == 6 && (input_dim == 4 || input_dim == 5)),
      "only support pad1D pad2D and pad3D now!");
}

void PadCall(Tensor& output, const Tensor& input, Pad& op) {
  c10::musa::MUSAGuard device_guard(input.device());
  muHandle& h = GetMudnnHandle();
  auto input_m = CreateMUTensor(input);
  auto output_m = CreateMUTensor(output);
  CHECK_MUDNN_STATUS(op.Run(h, output_m, input_m), "Run");
}

Tensor PadInternal(const Tensor& input, IntArrayRef pad, Pad& op) {
  auto input_sizes = input.sizes();
  auto l_inp = input.dim();
  auto l_pad = pad.size() / 2;
  auto l_diff = l_inp - l_pad;

  std::vector<int64_t> output_shape;
  for (size_t i = 0; i < static_cast<size_t>(l_diff); ++i) {
    output_shape.emplace_back(input_sizes[i]);
  }

  for (const auto i : c10::irange(static_cast<size_t>(l_pad))) {
    auto pad_idx = pad.size() - ((i + 1) * 2);
    auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
    TORCH_CHECK(
        new_dim > 0,
        "The input size ",
        input_sizes[l_diff + i],
        ", plus negative padding ",
        pad[pad_idx],
        " and ",
        pad[pad_idx + 1],
        " resulted in a negative output size, "
        "which is invalid. Check dimension ",
        l_diff + i,
        " of your input.");
    output_shape.emplace_back(new_dim);
  }
  auto output = at::empty(
      output_shape,
      input.options()
          .dtype(input.scalar_type())
          .memory_format(input.suggest_memory_format()));

  bool need_unsqueeze = false;
  if ((l_pad == 2 && l_inp == 2) || (l_pad == 3 && l_inp == 3)) {
    need_unsqueeze = true;
  }
  auto input_ = need_unsqueeze ? input.unsqueeze(0) : input;
  output = need_unsqueeze ? output.unsqueeze(0) : output;
  PadCall(output, input_, op);
  output = need_unsqueeze ? output.squeeze(0) : output;
  return output;
}

Tensor& ReflectPad1DOut(const Tensor& self, IntArrayRef pad, Tensor& output) {
  MUSA_TENSOR_TYPE_CHECK(self);
  MUSA_TENSOR_TYPE_CHECK(output);
  Pad op;
  ConfigPad(op, pad, Pad_MODE::REFLECT);
  TORCH_CHECK(output.is_contiguous(), "check contiguous failed");
  auto contiguous_self = self.contiguous();
  PadCall(output, contiguous_self, op);
  return output;
}

Tensor ReflectPad1D(const Tensor& self, IntArrayRef pad) {
  MUSA_TENSOR_TYPE_CHECK(self);
  Pad op;
  ConfigPad(op, pad, Pad_MODE::REFLECT);
  auto contiguous_self = self.contiguous();
  return PadInternal(contiguous_self, pad, op);
}

} // namespace musa
} // namespace at
