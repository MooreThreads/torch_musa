#include <ATen/Config.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#include <ATen/ops/embedding_backward_native.h>
#include <ATen/ops/embedding_dense_backward.h>
#include <ATen/ops/embedding_dense_backward_native.h>
#include <ATen/ops/embedding_native.h>
#include <ATen/ops/embedding_renorm_native.h>
#include <ATen/ops/embedding_sparse_backward.h>
#include <ATen/ops/embedding_sparse_backward_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

#include "torch_musa/csrc/aten/ops/Embedding.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

Tensor Embedding(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  TORCH_CHECK(weight.dim() == 2, "'weight' must be 2-D");
  auto indices_arg = TensorArg(indices, "indices", 1);
  auto weight_arg = TensorArg(weight, "weight", 1);
  checkScalarTypes("embedding", indices_arg, {kLong, kInt});
  checkScalarTypes("embedding", weight_arg, {kFloat, kHalf});
  // these varibs are not used in musa so far.
  UNUSED(scale_grad_by_freq);
  UNUSED(sparse);
  c10::musa::MUSAGuard device_guard(weight.device());

  auto size = indices.sizes().vec();
  for (auto d : weight.sizes().slice(1)) {
    size.push_back(d);
  }
  auto output = at::empty(
      size, weight.options().memory_format(at::MemoryFormat::Contiguous));

  Tensor weight_ = Contiguous(weight);
  Tensor indices_ = Contiguous(indices);

  EmbeddingRun(output, weight_, indices_, padding_idx);
  return output;
}

Tensor EmbeddingDenseBwd(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  TORCH_CHECK(
      grad_output.device().type() == kMUSA,
      "Device of grad_output tensor of embedding_dense_backward must "
      "be MUSA, but now is ",
      grad_output.device());
  TORCH_CHECK(
      indices.device().type() == kMUSA,
      "Device of indices tensor of embedding_dense_backward must be "
      "MUSA, but now is ",
      indices.device());
  TORCH_CHECK(
      grad_output.scalar_type() == at::ScalarType::Float,
      "Dtype of grad_output tensor of embedding_dense_backward only "
      "support Float32, but now is ",
      indices.scalar_type());
  TORCH_CHECK(
      indices.scalar_type() == at::ScalarType::Int ||
          indices.scalar_type() == at::ScalarType::Long,
      "Dtype of indices tensor of embedding_dense_backward only "
      "support Int/Long, but now is ",
      indices.scalar_type());
  // its not be used in muDNN so far.
  UNUSED(scale_grad_by_freq);
  c10::musa::MUSAGuard device_guard(grad_output.device());

  Tensor grad_input = at::empty(
      {num_weights, grad_output.size(-1)},
      grad_output.options().memory_format(at::MemoryFormat::Contiguous));
  auto contiguous_grad_output = Contiguous(grad_output);
  auto contiguous_indices = Contiguous(indices);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Embedding embedding;
  auto mt_grad_output = CreateMUTensor(contiguous_grad_output);
  auto mt_indices = CreateMUTensor(contiguous_indices);
  auto mt_grad_input = CreateMUTensor(grad_input);
  CHECK_MUDNN_STATUS(embedding.SetPaddingIdx(padding_idx), "SetPaddingIdx");
  CHECK_MUDNN_STATUS(
      embedding.RunDenseBwd(
          h, mt_grad_input, mt_grad_output, mt_indices, InternalMemAlloc),
      "Run");
  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("embedding", &Embedding);
  m.impl("embedding_dense_backward", &EmbeddingDenseBwd);
}

} // namespace musa
} // namespace at
