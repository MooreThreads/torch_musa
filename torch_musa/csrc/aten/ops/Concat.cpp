#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

inline bool ShouldSkipTensor(const Tensor& t) {
  return t.numel() == 0 && t.dim() == 1;
}

// these two utilities are borrowed from
// pytorch/aten/src/ATen/native/TensorShape.cpp
inline void CatCheckNoZeroDim(const MaterializedITensorListRef& tensors) {
  size_t i = 0;
  for (const Tensor& t : tensors) {
    TORCH_CHECK(
        t.dim() > 0,
        "zero-dimensional tensor (at position ",
        i,
        ") cannot be concatenated");
    i++;
  }
}

inline c10::MemoryFormat CatComputeOutputMemoryFormat(
    const MaterializedITensorListRef& inputs) {
  c10::optional<c10::MemoryFormat> format = c10::nullopt;
  for (const Tensor& t : inputs) {
    auto f = t.suggest_memory_format();
    if (f == c10::MemoryFormat::Contiguous) {
      return f;
    }
    if (format.has_value() && format.value() != f) {
      return c10::MemoryFormat::Contiguous;
    }
    format = f;
  }
  return format.value();
}

Tensor& CatOut(const at::ITensorListRef& tensors, int64_t dim, Tensor& out) {
  if (out.numel() == 0) {
    return out;
  }
  const auto& materialized = tensors.materialize();
  const OptionalDeviceGuard device_guard(device_of(materialized[0].get()));
  auto ref_type = at::native::result_type(materialized);
  auto memory_format = CatComputeOutputMemoryFormat(materialized);
  TORCH_CHECK(
      out.suggest_memory_format() == memory_format,
      "out tensor of cat.out should be in memory format of ",
      memory_format,
      "but now is ",
      out.suggest_memory_format());
  TORCH_CHECK(
      ref_type == out.scalar_type(),
      "out tensor dtype (",
      out.scalar_type(),
      ") should be same as ref tensor (",
      ref_type,
      ")");

  // Sicne muDNN concat doesn't support uncontiguous tensors,
  // so we store contiguous tensors for muTensors
  std::vector<Tensor> rt_tensors;
  int elements = 0;

  for (int idx = 0; idx < materialized.size(); ++idx) {
    if (materialized[idx].get().numel() > 0) {
      rt_tensors.emplace_back(
          FormatContiguous(materialized[idx].get(), memory_format)
              .to(ref_type));
      elements++;
    }
  }

  // Computational muTensors
  std::vector<at::musa::muTensor> mu_tensors;
  mu_tensors.reserve(elements);
  for (const auto& tensor : rt_tensors) {
    mu_tensors.emplace_back(at::musa::CreateMUTensor(tensor));
  }

  at::musa::muTensor out_ = at::musa::CreateMUTensor(out);
  at::musa::muHandle& h = at::GetMudnnHandle();
  ::musa::dnn::Concat op;

  int axis = dim;
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    if (axis == 1) {
      axis = 3;
    } else if (axis > 1) {
      --axis;
    }
  } else if (memory_format == at::MemoryFormat::ChannelsLast3d) {
    if (axis == 1) {
      axis = 4;
    } else if (axis > 1) {
      --axis;
    }
  }
  CHECK_MUDNN_STATUS(op.SetAxis(axis), "Set concat axis");

  CHECK_MUDNN_STATUS(
      op.Run(h, out_, elements, mu_tensors.data()), "Run Concat");

  return out;
}

Tensor Cat(const at::ITensorListRef& tensors, int64_t dim) {
  const auto& materialized = tensors.materialize();
  size_t valid = materialized.size();
  for (const auto i : c10::irange(materialized.size())) {
    if (!ShouldSkipTensor(materialized[i].get())) {
      valid = i;
      break;
    }
  }
  if (valid == materialized.size()) {
    // all tensors are invalid, early return
    return at::empty_like(materialized[0].get());
  }

  // first valid tensor
  const Tensor& ref = materialized[valid].get();

  CatCheckNoZeroDim(materialized);
  dim = at::legacy_cat_wrap_dim(dim, materialized);
  TORCH_CHECK(dim >= 0 && dim < ref.dim(), "Wrong Cat dim: ", dim);
  TORCH_CHECK(
      !materialized.empty(), "torch.cat(): expect a non-empty list of tensors");

  // Compute the output's shape
  std::vector<int64_t> output_shape{ref.sizes().vec()};
  output_shape[dim] = 0;
  for (const auto i : c10::irange(materialized.size())) {
    const Tensor& t = materialized[i];
    if (!ShouldSkipTensor(t)) {
      at::native::check_cat_shape_except_dim(materialized[valid], t, dim, i);
      output_shape[dim] += t.size(dim);
    }
  }
  // Compute the output's dtype and memory_format
  auto out_dtype = at::native::result_type(tensors);
  auto memory_format = CatComputeOutputMemoryFormat(materialized);

  TensorOptions options =
      ref.options().memory_format(memory_format).dtype(out_dtype);
  Tensor output = at::empty(output_shape, options);
  CatOut(tensors, dim, output);

  return output;
}

} // namespace musa
} // namespace at
