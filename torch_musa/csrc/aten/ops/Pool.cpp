#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace native {
namespace musa {

struct PoolParams {
  // [T, H, W] for 3d and [H, W] for 2d
  int k[3];
  int d[3];
  int pad[3];
  int dil[3];
  ::musa::dnn::Pooling::Mode mode;
  c10::optional<int64_t> divisor_override;
};

void PoolCall(
    const Tensor& input,
    const PoolParams& p,
    Tensor& output,
    Tensor* indices = nullptr) {
  auto out = CreateMUTensor(output);
  auto in = CreateMUTensor(input);
  muTensor inds;
  if (indices != nullptr) {
    inds = CreateMUTensor(*indices);
  }

  auto contiguous_input = input;

  ConfigFormat(contiguous_input, in, true);
  ConfigFormat(output, out, true);

  muHandle h;
  ::musa::dnn::Pooling pool;
  CHECK_MUDNN_STATUS(pool.SetMode(p.mode), "SetMode");
  CHECK_MUDNN_STATUS(
      pool.SetNdInfo(
          {p.k[0], p.k[1]},
          {p.pad[0], p.pad[1]},
          {p.d[0], p.d[1]},
          {p.dil[0], p.dil[1]}),
      "SetNdInfo");
  if (p.divisor_override.has_value()) {
    CHECK_MUDNN_STATUS(
        pool.SetDivisor(
            safe_downcast<int, int64_t>(p.divisor_override.value())),
        "SetDivisor");
  }
  CHECK_MUDNN_STATUS(pool.Run(h, out, in, inds), "Run");
}

void PoolCallBwd(
    const Tensor& grad_output,
    const PoolParams& p,
    Tensor& grad_input,
    const Tensor* indices = nullptr) {
  auto in = CreateMUTensor(grad_output);
  auto out = CreateMUTensor(grad_input);
  muTensor inds;
  if (indices) {
    inds = CreateMUTensor(*indices);
  }
  auto contiguous_input = grad_output;
  ConfigFormat(contiguous_input, in, true);
  ConfigFormat(grad_input, out, true);

  muHandle h;
  ::musa::dnn::Pooling pool;
  CHECK_MUDNN_STATUS(pool.SetMode(p.mode), "SetMode");
  CHECK_MUDNN_STATUS(
      pool.SetNdInfo(
          {p.k[0], p.k[1]},
          {p.pad[0], p.pad[1]},
          {p.d[0], p.d[1]},
          {p.dil[0], p.dil[1]}),
      "SetNdInfo");
  if (p.divisor_override.has_value()) {
    CHECK_MUDNN_STATUS(
        pool.SetDivisor(
            safe_downcast<int, int64_t>(p.divisor_override.value())),
        "SetDivisor");
  }
  CHECK_MUDNN_STATUS(pool.RunBwd(h, out, in, inds), "Run");
}

void AdaptiveAvgPool2dCheck(const Tensor& input, IntArrayRef output_size) {
  // Copy from AdaptiveAveragePooling.cpp
  TORCH_CHECK(
      output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
  int64_t ndim = input.ndimension();
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_avg_pool2d(): Expected input to have non-zero size "
        "for non-batch dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }
  TORCH_CHECK(
      (ndim == 3 || ndim == 4),
      "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got ",
      input.sizes());
}

void AdaptiveAvgPool2dInternal(
    const Tensor& input,
    const PoolParams& p,
    Tensor& output,
    IntArrayRef output_size) {
  AdaptiveAvgPool2dCheck(input, output_size);
  int64_t channels = input.size(-3);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  const auto memory_format = input.suggest_memory_format();
  auto options =
      input.options().dtype(input.scalar_type()).memory_format(memory_format);
  int64_t ndim = input.ndimension();

  if (ndim == 3) {
    output = at::empty({channels, output_height, output_width}, options);
  } else {
    int64_t nbatch = input.size(0);
    output =
        at::empty({nbatch, channels, output_height, output_width}, options);
  }
  if (output.numel() == 0) {
    return;
  }
  PoolCall(input, p, output, nullptr);
}

Tensor AdaptiveAvgPool2d(const Tensor& input, IntArrayRef output_size) {
  PoolParams params;
  params.mode = ::musa::dnn::Pooling::Mode::ADAPTIVE_AVGPOOL;
  Tensor output;
  AdaptiveAvgPool2dInternal(input, params, output, output_size);
  return output;
}

Tensor& AdaptiveAvgPool2dOut(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  PoolParams params;
  params.mode = ::musa::dnn::Pooling::Mode::ADAPTIVE_AVGPOOL;
  AdaptiveAvgPool2dCheck(input, output_size);
  TORCH_CHECK(
      input.dtype() == output.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `output` but got dtype ",
      output.dtype());
  CheckContiguous(output);
  PoolCall(input, params, output, nullptr);
  return output;
}

void AvgPool2dConfigParams(
    PoolParams& p,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  // Copy from AveragePool2d.cpp
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints");
  int64_t kH = kernel_size[0];
  int64_t kW = kernel_size.size() == 1 ? kH : kernel_size[1];

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints");
  int64_t dH = stride.empty() ? kH : stride[0];
  int64_t dW = stride.empty() ? kW : stride.size() == 1 ? dH : stride[1];

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of "
      "two ints");
  int64_t padH = padding[0];
  int64_t padW = padding.size() == 1 ? padH : padding[1];

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");
  p.divisor_override = divisor_override;

  p.k[0] = safe_downcast<int, int64_t>(kH);
  p.k[1] = safe_downcast<int, int64_t>(kW);
  p.pad[0] = safe_downcast<int, int64_t>(padH);
  p.pad[1] = safe_downcast<int, int64_t>(padW);
  p.d[0] = safe_downcast<int, int64_t>(dH);
  p.d[1] = safe_downcast<int, int64_t>(dW);
  p.dil[0] = 1;
  p.dil[1] = 1;
  p.mode = count_include_pad
      ? ::musa::dnn::Pooling::Mode::AVGPOOL_COUNT_PAD
      : ::musa::dnn::Pooling::Mode::AVGPOOL_COUNT_WITHOUT_PAD;
}

void AvgPool2dInternal(
    const Tensor& input,
    bool ceil_mode,
    PoolParams p,
    Tensor& output) {
  int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  int64_t n_input_plane = input.size(-3);
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_height = pooling_output_shape<int64_t>(
      input_height, p.k[0], p.pad[0], p.d[0], 1, ceil_mode);
  int64_t output_width = pooling_output_shape<int64_t>(
      input_width, p.k[1], p.pad[1], p.d[1], 1, ceil_mode);
  auto memory_format = input.suggest_memory_format();
  pool2d_shape_check(
      input,
      p.k[0],
      p.k[1],
      p.d[0],
      p.d[1],
      p.pad[0],
      p.pad[1],
      1,
      1,
      n_input_plane,
      input_height,
      input_width,
      output_height,
      output_width,
      memory_format);
  auto options =
      input.options().dtype(input.scalar_type()).memory_format(memory_format);

  if (input.ndimension() == 3) {
    output = at::empty({n_input_plane, output_height, output_width}, options);
  } else {
    output = at::empty(
        {nbatch, n_input_plane, output_height, output_width}, options);
  }
  PoolCall(input, p, output, nullptr);
}

void MaxPool2dConfigParams(
    PoolParams& p,
    IntArrayRef ker,
    IntArrayRef str,
    IntArrayRef pad,
    IntArrayRef dil) {
  // Copy from DilatedMaxPool2d.cpp : max_pool2d_with_indices
  TORCH_CHECK(
      ker.size() == 1 || ker.size() == 2,
      "max_pool2d: ker must either be a single int, or a tuple of two ints")
  p.k[0] = safe_downcast<int, int64_t>(ker[0]);
  p.k[1] = ker.size() == 1 ? p.k[0] : safe_downcast<int, int64_t>(ker[1]);

  TORCH_CHECK(
      str.size() == 0 || str.size() == 1 || str.size() == 2,
      "max_pool2d: str must either be omitted, a single int, or a "
      "tuple of two ints")
  p.d[0] = str.empty() ? p.k[0] : safe_downcast<int, int64_t>(str[0]);
  p.d[1] = str.empty()  ? p.k[1]
      : str.size() == 1 ? p.d[0]
                        : safe_downcast<int, int64_t>(str[1]);

  TORCH_CHECK(
      pad.size() == 1 || pad.size() == 2,
      "max_pool2d: pad must be either be a single int, or a tuple of two ints");
  p.pad[0] = safe_downcast<int, int64_t>(pad[0]);
  p.pad[1] = pad.size() == 1 ? p.pad[0] : safe_downcast<int, int64_t>(pad[1]);

  TORCH_CHECK(
      dil.size() == 1 || dil.size() == 2,
      "max_pool2d: dil must be either a single int, or a tuple of two ints");
  p.dil[0] = safe_downcast<int, int64_t>(dil[0]);
  p.dil[1] = dil.size() == 1 ? p.dil[0] : safe_downcast<int, int64_t>(dil[1]);
  p.mode = ::musa::dnn::Pooling::Mode::MAXPOOL;
}

void MaxPool2dInternal(
    const Tensor& input,
    const PoolParams& p,
    bool ceil_mode,
    Tensor& output,
    Tensor* indices) {
  /* sizes */
  bool add_batch_dimension = input.ndimension() == 4 ? false : true;
  auto contiguous_input = add_batch_dimension ? input.unsqueeze(0) : input;
  int64_t nbatch = contiguous_input.size(-4);
  int64_t n_input_plane = contiguous_input.size(-3);
  int64_t inH = contiguous_input.size(-2);
  int64_t inW = contiguous_input.size(-1);

  int64_t output_height = pooling_output_shape<int64_t>(
      inH, p.k[0], p.pad[0], p.d[0], p.dil[0], ceil_mode);
  int64_t output_width = pooling_output_shape<int64_t>(
      inW, p.k[1], p.pad[1], p.d[1], p.dil[1], ceil_mode);

  // Our own code
  const auto memory_format = input.suggest_memory_format();
  auto options = contiguous_input.options()
                     .dtype(contiguous_input.scalar_type())
                     .memory_format(memory_format);
  DimnameList maybe_names = input.has_names() ? input.names() : DimnameList{};
  if (add_batch_dimension) {
    output =
        at::empty({1, n_input_plane, output_height, output_width}, options);
    if (indices != nullptr) {
      *indices = at::empty(
          {1, n_input_plane, output_height, output_width},
          options.dtype(kLong));
    }
  } else {
    output = at::empty(
        {nbatch, n_input_plane, output_height, output_width}, options);
    if (indices != nullptr) {
      *indices = at::empty(
          {nbatch, n_input_plane, output_height, output_width},
          options.dtype(kLong));
    }
  }
  if (!maybe_names.empty()) {
    namedinference::propagate_names(output, maybe_names);
    if (indices != nullptr) {
      namedinference::propagate_names(*indices, maybe_names);
    }
  }
  PoolCall(contiguous_input, p, output, indices);
  if (add_batch_dimension) {
    output = output.squeeze(0);
  }
}

void MaxPool2dInternalBwd(
    const Tensor& grad_output,
    const PoolParams& p,
    bool ceil_mode,
    const Tensor& input,
    Tensor& grad_input,
    const Tensor& indices) {
  if (ceil_mode) {
    C10_LOG_FIRST_N(WARNING, 1)
        << "ceil_mode is not use in MaxPool2dInternalBwd";
  }
  /* sizes */
  bool add_batch_dimension = input.ndimension() == 4 ? false : true;
  auto contiguous_grad_output =
      add_batch_dimension ? grad_output.unsqueeze(0) : grad_output;
  int64_t nbatch = add_batch_dimension ? 1 : input.size(-4);
  int64_t n_input_plane = input.size(-3);
  int64_t inH = input.size(-2);
  int64_t inW = input.size(-1);

  // Our own code
  const auto memory_format = input.suggest_memory_format();
  auto options =
      input.options().dtype(input.scalar_type()).memory_format(memory_format);
  DimnameList maybe_names = input.has_names() ? input.names() : DimnameList{};
  if (add_batch_dimension) {
    grad_input = at::empty({1, n_input_plane, inH, inW}, options);
  } else {
    grad_input = at::empty({nbatch, n_input_plane, inH, inW}, options);
  }
  if (!maybe_names.empty()) {
    namedinference::propagate_names(grad_input, maybe_names);
  }
  PoolCallBwd(contiguous_grad_output, p, grad_input, &indices);
  if (add_batch_dimension) {
    grad_input = grad_input.squeeze(0);
  }
}

std::tuple<Tensor, Tensor> MaxPool2dIndices(
    const Tensor& input,
    IntArrayRef ker,
    IntArrayRef str,
    IntArrayRef pad,
    IntArrayRef dil,
    bool ceil_mode) {
  PoolParams params;
  MaxPool2dConfigParams(params, ker, str, pad, dil);
  auto r = std::make_tuple(Tensor(), Tensor());
  MaxPool2dInternal(input, params, ceil_mode, std::get<0>(r), &std::get<1>(r));
  return r;
}

Tensor MaxPool2dIndicesBwd(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef ker,
    IntArrayRef str,
    IntArrayRef pad,
    IntArrayRef dil,
    bool ceil_mode,
    const Tensor& indices) {
  PoolParams params;
  MaxPool2dConfigParams(params, ker, str, pad, dil);
  Tensor grad_input;
  MaxPool2dInternalBwd(
      grad_output, params, ceil_mode, self, grad_input, indices);
  return grad_input;
}

std::tuple<Tensor&, Tensor&> MaxPool2dIndicesOut(
    const Tensor& input,
    IntArrayRef ker,
    IntArrayRef str,
    IntArrayRef pad,
    IntArrayRef dil,
    bool ceil_mode,
    Tensor& output,
    Tensor& indices) {
  if (ceil_mode) {
    C10_LOG_FIRST_N(WARNING, 1)
        << "ceil mode is invalid in MaxPool2dIndicesOut";
  }
  PoolParams params;
  MaxPool2dConfigParams(params, ker, str, pad, dil);
  CheckContiguous(output);
  CheckContiguous(indices);
  PoolCall(input, params, output, &indices);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

Tensor& MaxPool2dIndicesBwdOut(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef ker,
    IntArrayRef str,
    IntArrayRef pad,
    IntArrayRef dil,
    bool ceil_mode,
    const Tensor& indices,
    Tensor& grad_input) {
  if (ceil_mode) {
    C10_LOG_FIRST_N(WARNING, 1)
        << "ceil_mode is invalid in MaxPool2dIndicesBwdOut";
  }
  PoolParams params;
  MaxPool2dConfigParams(params, ker, str, pad, dil);
  CheckContiguous(grad_input);
  grad_input.zero_();
  PoolCallBwd(grad_output, params, grad_input, &indices);
  return grad_input;
}

Tensor AvgPool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor output;
  PoolParams params;
  AvgPool2dConfigParams(
      params,
      kernel_size,
      stride,
      padding,
      count_include_pad,
      divisor_override);
  AvgPool2dInternal(input, ceil_mode, params, output);
  return output;
}

Tensor& AvgPool2dOut(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& output) {
  if (ceil_mode) {
    C10_LOG_FIRST_N(WARNING, 1) << "ceil_mode is invalid in AvgPool2dOut";
  }
  PoolParams params;
  AvgPool2dConfigParams(
      params,
      kernel_size,
      stride,
      padding,
      count_include_pad,
      divisor_override);
  CheckContiguous(output);
  PoolCall(input, params, output, nullptr);
  return output;
}

Tensor& AvgPool2dOutBwd(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& grad_input) {
  TORCH_CHECK(
      grad_output.device().type() == kMUSA,
      "Device of grad_output tensor of AvgPool2dBackward must be MUSA, ",
      "but now is ",
      grad_output.device());
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of AvgPool2dBackward must be MUSA, but now is ",
      input.device());
  TORCH_CHECK(
      grad_input.device().type() == kMUSA,
      "Device of grad_input tensor of AvgPool2dBackward must be MUSA, "
      "but now is ",
      grad_input.device());
  TORCH_CHECK(
      grad_output.scalar_type() == at::ScalarType::Float,
      "Dtype of grad_output tensor of AvgPool2dBackward only support Float32, ",
      "but now it is ",
      grad_output.scalar_type());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of AvgPool2dBackward only support Float32, ",
      "but now it is ",
      input.scalar_type());

  PoolParams params;
  AvgPool2dConfigParams(
      params,
      kernel_size,
      stride,
      padding,
      count_include_pad,
      divisor_override);

  int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  int64_t n_input_plane = input.size(-3);
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_height = pooling_output_shape<int64_t>(
      input_height, params.k[0], params.pad[0], params.d[0], 1, ceil_mode);
  int64_t output_width = pooling_output_shape<int64_t>(
      input_width, params.k[1], params.pad[1], params.d[1], 1, ceil_mode);
  auto memory_format = input.suggest_memory_format();
  avg_pool2d_backward_shape_check(
      input,
      grad_output,
      nbatch,
      params.k[0],
      params.k[1],
      params.d[0],
      params.d[1],
      params.pad[0],
      params.pad[1],
      n_input_plane,
      input_height,
      input_width,
      output_height,
      output_width,
      memory_format);

  if (input.ndimension() == 3) {
    grad_input.resize_({n_input_plane, input_height, input_width});
  } else {
    grad_input.resize_({nbatch, n_input_plane, input_height, input_width});
  }
  PoolCallBwd(grad_output, params, grad_input, nullptr);
  return grad_input;
}

Tensor AvgPool2dBwd(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor grad_input = at::empty(input.sizes(), input.options());
  auto result = AvgPool2dOutBwd(
      grad_output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override,
      grad_input);
  (void)result;
  return grad_input;
}

Tensor AdaptiveAvgPool2dBwd(const Tensor& grad_output, const Tensor& input) {
  PoolParams params;
  params.mode = ::musa::dnn::Pooling::Mode::ADAPTIVE_AVGPOOL;

  Tensor grad_input = at::empty(input.sizes(), input.options());

  TORCH_CHECK(
      grad_output.device().type() == kMUSA,
      "Device of grad_output tensor of AvgPool2dBackward must be MUSA, ",
      "but now is ",
      grad_output.device());
  TORCH_CHECK(
      input.device().type() == kMUSA,
      "Device of input tensor of AvgPool2dBackward must be MUSA, but now is ",
      input.device());
  TORCH_CHECK(
      grad_output.scalar_type() == at::ScalarType::Float,
      "Dtype of grad_output tensor of AvgPool2dBackward only support Float32, ",
      "but now it is ",
      grad_output.scalar_type());
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of AvgPool2dBackward only support Float32, ",
      "but now it is ",
      input.scalar_type());
  PoolCallBwd(grad_output, params, grad_input, nullptr);
  return grad_input;
}


TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_adaptive_avg_pool2d", &AdaptiveAvgPool2d);
  m.impl("adaptive_avg_pool2d.out", &AdaptiveAvgPool2dOut);
  m.impl("_adaptive_avg_pool2d_backward", &AdaptiveAvgPool2dBwd);

  m.impl("avg_pool2d", &AvgPool2d);
  m.impl("avg_pool2d.out", &AvgPool2dOut);
  m.impl("avg_pool2d_backward", AvgPool2dBwd);
  m.impl("avg_pool2d_backward.grad_input", AvgPool2dOutBwd);

  m.impl("max_pool2d_with_indices", &MaxPool2dIndices);
  m.impl("max_pool2d_with_indices_backward", &MaxPool2dIndicesBwd);
  m.impl("max_pool2d_with_indices.out", &MaxPool2dIndicesOut);
  m.impl("max_pool2d_with_indices_backward_out", &MaxPool2dIndicesBwdOut);
}

} // namespace musa
} // namespace native
} // namespace at
