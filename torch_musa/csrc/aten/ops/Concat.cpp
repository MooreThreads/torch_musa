#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace native {
namespace musa {

inline bool cat_should_skip_tensor(const Tensor& t) {
  return t.numel() == 0 && t.dim() == 1;
}

// Check to see if the shape of tensors is compatible
// for being concatenated along a given dimension.
inline void check_cat_shape_except_dim(
    const Tensor& first,
    const Tensor& second,
    int64_t dimension) {
  int64_t first_dims = first.dim();
  int64_t second_dims = second.dim();
  TORCH_CHECK(
      first_dims == second_dims,
      "Tensors must have same number of dimensions: got ",
      first_dims,
      " and ",
      second_dims);
  for (const auto dim : c10::irange(first_dims)) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = first.sizes()[dim];
    int64_t second_dim_size = second.sizes()[dim];
    TORCH_CHECK(
        first_dim_size == second_dim_size,
        "Sizes of tensors must match except in dimension ",
        dimension,
        ". Expected size ",
        first_dim_size,
        " but got size ",
        second_dim_size,
        " in the list.");
  }
}

void ConcatImpl(Tensor& output, int dim, TensorList tensors) {
  int64_t num_inputs = tensors.size();
  std::vector<muTensor> musa_tensors;
  musa_tensors.reserve(num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    // Skip the tensor[i] when cat_dim is empty,
    if (tensors[i].size(dim) == 0) {
      continue;
    }
    muTensor mt_tensor = CreateMUTensor(tensors[i]);
    musa_tensors.emplace_back(mt_tensor);
  }
  if (musa_tensors.size() == 0) {
    return;
  }
  auto om = CreateMUTensor(output);
  ::musa::dnn::Handle handle;
  ::musa::dnn::Concat op;
  CHECK_MUDNN_STATUS(op.SetAxis(dim), "SetAxis");
  CHECK_MUDNN_STATUS(
      op.Run(handle, om, musa_tensors.size(), musa_tensors.data()), "Run");
}

Tensor Cat(const at::ITensorListRef& tensors, int64_t dim = 0) {
  Tensor valid_tensor;
  auto num_inputs = tensors.size();
  for (const auto& t : tensors) {
    if (!cat_should_skip_tensor(t)) {
      valid_tensor = t;
      break;
    }
  }

  // cal output_sizes must use valid_id !!!
  auto ndim = valid_tensor.dim();
  dim = ((dim % ndim) + ndim) % ndim;
  int64_t dim_out_size = 0;
  std::vector<Tensor> valid_tensors;
  valid_tensors.reserve(num_inputs);

  for (const auto& t : tensors) {
    if (!cat_should_skip_tensor(t)) {
      check_cat_shape_except_dim(valid_tensor, t, dim);
      dim_out_size = dim_out_size + t.size(dim);
      Tensor one = Contiguous(t);
      valid_tensors.emplace_back(one);
    }
  }

  auto output_sizes = valid_tensor.sizes().vec();
  output_sizes[dim] = dim_out_size;

  Tensor output = empty_mtgpu(
      output_sizes,
      valid_tensor.scalar_type(),
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  ConcatImpl(output, dim, valid_tensors);
  return output;
}

Tensor& CatOut(const at::ITensorListRef& tensors, int64_t dim, Tensor& output) {
  auto num_inputs = tensors.size();
  Tensor valid_tensor;
  for (const auto& t : tensors) {
    if (!cat_should_skip_tensor(t)) {
      valid_tensor = t;
      break;
    }
  }
  auto ndim = valid_tensor.dim();
  dim = ((dim % ndim) + ndim) % ndim;

  std::vector<Tensor> valid_tensors;
  valid_tensors.reserve(num_inputs);

  for (const auto& t : tensors) {
    if (!cat_should_skip_tensor(t)) {
      check_cat_shape_except_dim(valid_tensor, t, dim);
      Tensor one = Contiguous(t);
      valid_tensors.emplace_back(one);
    }
  }
  ConcatImpl(output, dim, valid_tensors);
  return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("cat", &Cat);
  m.impl("_cat", &Cat);
  m.impl("cat.out", &CatOut);
}

} // namespace musa
} // namespace native
} // namespace at
