#include <ATen/TensorMeta.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/adaptive_max_pool2d_backward_native.h>
#include <ATen/ops/adaptive_max_pool2d_native.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {
using PoolingMode = ::musa::dnn::Pooling::Mode;

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
