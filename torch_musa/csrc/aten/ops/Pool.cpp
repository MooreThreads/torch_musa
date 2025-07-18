#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

using PoolingMode = ::musa::dnn::Pooling::Mode;

Tensor& AdaptiveAvgPool3DOutMUSA(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output);

Tensor AdaptiveAvgPool3DMUSA(const Tensor& input, IntArrayRef output_size);

Tensor& AdaptiveAvgPool3DBackwardOutMUSA(
    const Tensor& grad_output,
    const Tensor& input,
    Tensor& grad_input);

Tensor AdaptiveAvgPool3DBackwardMUSA(
    const Tensor& grad_output,
    const Tensor& input);

struct PoolParams {
  // [T, H, W] for 3d and [H, W] for 2d
  int k[3];
  int d[3];
  int pad[3];
  int dil[3];
  PoolingMode mode;
  c10::optional<int64_t> divisor_override;
};

void PoolCall(
    const Tensor& input,
    const PoolParams& p,
    Tensor& output,
    Tensor* indices = nullptr) {
  c10::musa::MUSAGuard device_guard(input.device());
  const auto output_memory_format = output.suggest_memory_format();
  auto contiguous_input = FormatContiguous(input, output_memory_format);
  auto out = CreateMUTensor(output);
  auto in = CreateMUTensor(contiguous_input);
  muTensor inds;
  if (indices != nullptr) {
    auto contiguous_indices = FormatContiguous(*indices, output_memory_format);
    inds = CreateMUTensor(contiguous_indices);
  }
  muHandle& h = GetMudnnHandle();
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
        pool.SetDivisor(at::native::safe_downcast<int, int64_t>(
            p.divisor_override.value())),
        "SetDivisor");
  }
  CHECK_MUDNN_STATUS(pool.Run(h, out, in, inds), "Run");
}

void PoolCallBwd(
    const Tensor& grad_output,
    const PoolParams& p,
    Tensor& grad_input,
    const Tensor* indices = nullptr) {
  c10::musa::MUSAGuard device_guard(grad_output.device());
  const auto grad_input_memory_format = grad_input.suggest_memory_format();
  auto contiguous_grad_output =
      FormatContiguous(grad_output, grad_input_memory_format);
  auto in = CreateMUTensor(contiguous_grad_output);
  auto out = CreateMUTensor(grad_input);
  muTensor inds;
  Tensor contiguous_indices;
  if (indices) {
    auto contiguous_indices =
        FormatContiguous(*indices, grad_input_memory_format);
    inds = CreateMUTensor(contiguous_indices);
  }

  muHandle& h = GetMudnnHandle();
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
        pool.SetDivisor(at::native::safe_downcast<int, int64_t>(
            p.divisor_override.value())),
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
  params.mode = PoolingMode::ADAPTIVE_AVGPOOL;
  Tensor output;
  AdaptiveAvgPool2dInternal(input, params, output, output_size);
  return output;
}

Tensor& AdaptiveAvgPool2dOut(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  PoolParams params;
  params.mode = PoolingMode::ADAPTIVE_AVGPOOL;
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

  p.k[0] = at::native::safe_downcast<int, int64_t>(kH);
  p.k[1] = at::native::safe_downcast<int, int64_t>(kW);
  p.pad[0] = at::native::safe_downcast<int, int64_t>(padH);
  p.pad[1] = at::native::safe_downcast<int, int64_t>(padW);
  p.d[0] = at::native::safe_downcast<int, int64_t>(dH);
  p.d[1] = at::native::safe_downcast<int, int64_t>(dW);
  p.dil[0] = 1;
  p.dil[1] = 1;
  p.mode = count_include_pad ? PoolingMode::AVGPOOL_COUNT_PAD
                             : PoolingMode::AVGPOOL_COUNT_WITHOUT_PAD;
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
  int64_t output_height = at::native::pooling_output_shape<int64_t>(
      input_height, p.k[0], p.pad[0], p.d[0], 1, ceil_mode);
  int64_t output_width = at::native::pooling_output_shape<int64_t>(
      input_width, p.k[1], p.pad[1], p.d[1], 1, ceil_mode);
  auto memory_format = input.suggest_memory_format();
  at::native::pool2d_shape_check(
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
  if (nbatch == 0) {
    output = at::empty(
        {nbatch, n_input_plane, output_height, output_width}, options);
    return;
  }
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
  p.k[0] = at::native::safe_downcast<int, int64_t>(ker[0]);
  p.k[1] = ker.size() == 1 ? p.k[0]
                           : at::native::safe_downcast<int, int64_t>(ker[1]);

  TORCH_CHECK(
      str.size() == 0 || str.size() == 1 || str.size() == 2,
      "max_pool2d: str must either be omitted, a single int, or a "
      "tuple of two ints")
  p.d[0] =
      str.empty() ? p.k[0] : at::native::safe_downcast<int, int64_t>(str[0]);
  p.d[1] = str.empty()  ? p.k[1]
      : str.size() == 1 ? p.d[0]
                        : at::native::safe_downcast<int, int64_t>(str[1]);

  TORCH_CHECK(
      pad.size() == 1 || pad.size() == 2,
      "max_pool2d: pad must be either be a single int, or a tuple of two ints");
  p.pad[0] = at::native::safe_downcast<int, int64_t>(pad[0]);
  p.pad[1] = pad.size() == 1 ? p.pad[0]
                             : at::native::safe_downcast<int, int64_t>(pad[1]);

  TORCH_CHECK(
      dil.size() == 1 || dil.size() == 2,
      "max_pool2d: dil must be either a single int, or a tuple of two ints");
  p.dil[0] = at::native::safe_downcast<int, int64_t>(dil[0]);
  p.dil[1] = dil.size() == 1 ? p.dil[0]
                             : at::native::safe_downcast<int, int64_t>(dil[1]);
  p.mode = PoolingMode::MAXPOOL;
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

  int64_t output_height = at::native::pooling_output_shape<int64_t>(
      inH, p.k[0], p.pad[0], p.d[0], p.dil[0], ceil_mode);
  int64_t output_width = at::native::pooling_output_shape<int64_t>(
      inW, p.k[1], p.pad[1], p.d[1], p.dil[1], ceil_mode);

  // Our own code
  const auto memory_format = contiguous_input.suggest_memory_format();
  auto options = contiguous_input.options()
                     .dtype(contiguous_input.scalar_type())
                     .memory_format(memory_format);
  DimnameList maybe_names = input.has_names() ? input.names() : DimnameList{};
  if (nbatch == 0) {
    output = at::empty(
        {nbatch, n_input_plane, output_height, output_width}, options);
    if (indices != nullptr) {
      *indices = at::empty(
          {nbatch, n_input_plane, output_height, output_width},
          options.dtype(kLong));
    }
    return;
  }
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
  c10::musa::MUSAGuard device_guard(input.device());
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
  int64_t output_height = at::native::pooling_output_shape<int64_t>(
      input_height, params.k[0], params.pad[0], params.d[0], 1, ceil_mode);
  int64_t output_width = at::native::pooling_output_shape<int64_t>(
      input_width, params.k[1], params.pad[1], params.d[1], 1, ceil_mode);
  auto memory_format = input.suggest_memory_format();
  at::native::avg_pool2d_backward_shape_check(
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
  if (at::musa::maybeDNNOpSupportBFloat16()) {
    TORCH_MUSA_CHECK_FLOATING_TYPES(
        input.scalar_type(), "AdaptiveAvgPool2dBwd");
  } else {
    TORCH_MUSA_CHECK_DTYPES(
        input.scalar_type(),
        "AdaptiveAvgPool2dBwd",
        at::ScalarType::Float,
        at::ScalarType::Half);
  }
  PoolParams params;
  params.mode = PoolingMode::ADAPTIVE_AVGPOOL;

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
  PoolCallBwd(grad_output, params, grad_input, nullptr);
  return grad_input;
}

::std::tuple<at::Tensor, at::Tensor> MaxPool3dIndices(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::max_pool3d_with_indices_cuda(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

::std::tuple<at::Tensor&, at::Tensor&> MaxPool3dIndicesOut(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    at::Tensor& out,
    at::Tensor& indices) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::max_pool3d_with_indices_out_cuda(
      self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
}

at::Tensor MaxPool3dIndicesBwd(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::max_pool3d_with_indices_backward_cuda(
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      indices);
}

at::Tensor& MaxPool3dIndicesBwdOut(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices,
    at::Tensor& grad_input) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::max_pool3d_with_indices_backward_out_cuda(
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      indices,
      grad_input);
}

Tensor& AdaptiveAvgPool3dOut(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::musa::AdaptiveAvgPool3DOutMUSA(input, output_size, output);
}

Tensor AdaptiveAvgPool3d(const Tensor& input, IntArrayRef output_size) {
  const OptionalDeviceGuard device_guard(device_of(input));
  return at::musa::AdaptiveAvgPool3DMUSA(input, output_size);
}

Tensor& AdaptiveAvgPool3dBackwardOut(
    const Tensor& grad_output,
    const Tensor& self,
    Tensor& grad_input) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::musa::AdaptiveAvgPool3DBackwardOutMUSA(
      grad_output, self, grad_input);
}

Tensor AdaptiveAvgPool3dBackward(
    const Tensor& grad_output,
    const Tensor& self) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::musa::AdaptiveAvgPool3DBackwardMUSA(grad_output, self);
}

TORCH_IMPL_FUNC(adaptive_max_pool2d_out_musa)
(const Tensor& input,
 IntArrayRef output_size,
 const Tensor& output,
 const Tensor& indices) {
  if (at::musa::maybeDNNOpSupportBFloat16()) {
    TORCH_MUSA_CHECK_FLOATING_TYPES(
        input.scalar_type(), "adaptive_max_pool2d_out_musa");
  } else {
    TORCH_MUSA_CHECK_DTYPES(
        input.scalar_type(),
        "adaptive_max_pool2d_out_musa",
        at::ScalarType::Float,
        at::ScalarType::Half);
  }
  TensorArg output_arg{output, "output", 1};
  TensorArg indices_arg{indices, "indices", 2};
  TensorArg input_arg{input, "input", 3};

  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});
  if (input.numel() == 0) {
    return;
  }

  c10::musa::MUSAGuard device_guard(input.device());
  at::MemoryFormat output_memory_format = output.suggest_memory_format();
  Tensor input_tmp = input.suggest_memory_format() == output_memory_format
      ? input
      : FormatContiguous(input, output_memory_format);
  auto input_mu = CreateMUTensor(input_tmp);
  auto output_mu = CreateMUTensor(output);
  auto indices_mu = CreateMUTensor(indices);
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Pooling pool;
  CHECK_MUDNN_STATUS(pool.SetMode(PoolingMode::ADAPTIVE_MAXPOOL), "SetMode");
  CHECK_MUDNN_STATUS(pool.Run(h, output_mu, input_mu, indices_mu), "Run");
}

TORCH_IMPL_FUNC(adaptive_max_pool2d_backward_out_musa)
(const Tensor& gradOutput,
 const Tensor& input,
 const Tensor& indices,
 const Tensor& gradInput) {
  if (at::musa::maybeDNNOpSupportBFloat16()) {
    TORCH_MUSA_CHECK_FLOATING_TYPES(
        input.scalar_type(), "adaptive_max_pool2d_backward_out_musa");
  } else {
    TORCH_MUSA_CHECK_DTYPES(
        input.scalar_type(),
        "adaptive_max_pool2d_backward_out_musa",
        at::ScalarType::Float,
        at::ScalarType::Half);
  }

  TensorArg grad_input_arg{gradInput, "gradInput", 1};
  TensorArg grad_output_arg{gradOutput, "gradOutput", 2};
  TensorArg input_arg{input, "input", 3};
  TensorArg indices_arg{indices, "indices", 4};

  checkAllSameGPU(
      __func__, {grad_input_arg, grad_output_arg, input_arg, indices_arg});

  if (gradOutput.numel() == 0) {
    return;
  }

  c10::musa::MUSAGuard device_guard(input.device());
  Tensor grad_output_tmp = Contiguous(gradOutput);
  Tensor indices_tmp = Contiguous(indices);
  Tensor grad_input_tmp = Contiguous(gradInput);
  auto grad_output_mu = CreateMUTensor(grad_output_tmp);
  auto grad_input_mu = CreateMUTensor(grad_input_tmp);
  auto indices_mu = CreateMUTensor(indices_tmp);
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Pooling pool;
  CHECK_MUDNN_STATUS(pool.SetMode(PoolingMode::ADAPTIVE_MAXPOOL), "SetMode");
  CHECK_MUDNN_STATUS(
      pool.RunBwd(h, grad_input_mu, grad_output_mu, indices_mu), "Run");

  if (!gradInput.is_contiguous() ||
      gradInput.is_contiguous(MemoryFormat::ChannelsLast)) {
    // (N, 1, H, W) and (N, C, 1, 1) cases also taken into consideration
    gradInput.copy_(grad_input_tmp);
  }
}
} // namespace musa
} // namespace at
