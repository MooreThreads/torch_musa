#include <ATen/Dispatch.h>
#include <ATen/native/Padding.h>
#include "ATen/TensorMeta.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/reflection_pad1d_native.h>
#include <ATen/ops/reflection_pad2d_native.h>
#include <ATen/ops/reflection_pad3d_native.h>
#include <ATen/ops/replication_pad1d_native.h>
#include <ATen/ops/replication_pad2d_native.h>
#include <ATen/ops/replication_pad3d_native.h>
#endif

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <mudnn.h>

#include <optional>
#include <vector>

namespace at {
namespace musa {
namespace {
using ::musa::dnn::Pad;

void PadImpl(
    Tensor& output,
    const Tensor& input,
    const IntArrayRef& padding_size,
    Pad::Mode mode,
    std::optional<double> constant_value) {
  TORCH_MUSA_CHECK_DTYPES(
      output.scalar_type(),
      "PadImpl",
      ScalarType::Float,
      ScalarType::Half,
      ScalarType::Int,
      ScalarType::Short,
      ScalarType::Char,
      ScalarType::Byte,
      ScalarType::Bool,
      ScalarType::BFloat16);

  if (input.numel() == 0) {
    return;
  }

  std::vector<int> pad;
  pad.reserve(padding_size.size());
  for (const auto& elem : padding_size) {
    pad.push_back(static_cast<int>(elem));
  }

  c10::musa::MUSAGuard device_guard(input.device());
  const auto output_memory_format = output.suggest_memory_format();
  auto contiguous_input = FormatContiguous(input, output_memory_format);
  auto out = CreateMUTensor(output);
  auto in = CreateMUTensor(contiguous_input);

  muHandle& h = GetMudnnHandle();
  Pad op;
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  if (constant_value.has_value()) {
    CHECK_MUDNN_STATUS(op.SetValue(constant_value.value()), "SetValue");
  }
  CHECK_MUDNN_STATUS(
      op.SetPaddingInfo(pad.size(), pad.data()), "SetPaddingInfo");
  CHECK_MUDNN_STATUS(op.Run(h, out, in), "RunPad");
}

} // anonymous namespace

TORCH_IMPL_FUNC(replication_pad1d_out_musa)
(const Tensor& input, IntArrayRef padding_size, const Tensor& output) {
  PadImpl(
      const_cast<Tensor&>(output),
      input,
      padding_size,
      Pad::Mode::REPLICATE,
      std::nullopt);
}

TORCH_IMPL_FUNC(replication_pad2d_out_musa)
(const Tensor& input, IntArrayRef padding_size, const Tensor& output) {
  PadImpl(
      const_cast<Tensor&>(output),
      input,
      padding_size,
      Pad::Mode::REPLICATE,
      std::nullopt);
}

TORCH_IMPL_FUNC(replication_pad3d_out_musa)
(const Tensor& input, IntArrayRef padding_size, const Tensor& output) {
  PadImpl(
      const_cast<Tensor&>(output),
      input,
      padding_size,
      Pad::Mode::REPLICATE,
      std::nullopt);
}

TORCH_IMPL_FUNC(reflection_pad1d_out_musa)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  PadImpl(
      const_cast<Tensor&>(output),
      input,
      padding,
      Pad::Mode::REFLECT,
      std::nullopt);
}

Tensor& ReflectionPad2DOutMUSA(
    const Tensor& input,
    IntArrayRef padding,
    Tensor& output) {
  // configure shape of output
  at::native::padding::check_valid_input<2>(input, padding);

  int dim_c = 0;
  int dim_h = 1;
  int dim_w = 2;
  int batch_size = 1;

  if (input.ndimension() == 4) {
    batch_size = input.size(0);
    dim_c++;
    dim_h++;
    dim_w++;
  }
  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int input_c = input.size(dim_c);
  int input_h = input.size(dim_h);
  int input_w = input.size(dim_w);

  TORCH_CHECK(
      pad_l < input_w && pad_r < input_w,
      "Padding size should be less than the corresponding input dimension, but "
      "got: padding (",
      pad_l,
      ", ",
      pad_r,
      ") at dimension ",
      dim_w,
      " of input ",
      input.sizes());
  TORCH_CHECK(
      pad_t < input_h && pad_b < input_h,
      "Padding size should be less than the corresponding input dimension, but "
      "got: padding (",
      pad_t,
      ", ",
      pad_b,
      ") at dimension ",
      dim_h,
      " of input ",
      input.sizes());

  int output_h = input_h + pad_t + pad_b;
  int output_w = input_w + pad_l + pad_r;

  if (input.ndimension() == 3) {
    output.resize_({input_c, output_h, output_w});
  } else {
    output.resize_({batch_size, input_c, output_h, output_w});
  }

  PadImpl(output, input, padding, Pad::Mode::REFLECT, std::nullopt);

  return output;
}

Tensor ReflectionPad2DMUSA(const Tensor& input, IntArrayRef padding) {
  // output will be resized later
  auto output = at::empty({0}, input.options());
  ReflectionPad2DOutMUSA(input, padding, output);
  return output;
}

TORCH_IMPL_FUNC(reflection_pad3d_out_musa)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  PadImpl(
      const_cast<Tensor&>(output),
      input,
      padding,
      Pad::Mode::REFLECT,
      std::nullopt);
}

} // namespace musa
} // namespace at
