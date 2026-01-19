#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cat_native.h>
#include <ATen/ops/empty.h>
#endif

#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at::musa {

extern void cat_complex_out_musa(
    const ITensorListRef& tensors,
    int64_t dim,
    int64_t valid,
    bool all_contiguous,
    bool all_same_dtype,
    bool all_same_sizes_and_stride,
    MemoryFormat memory_format,
    const Tensor& result);
namespace {

// Make tensor `c_type` and `memory_format` contiguous.
std::pair<Tensor, bool> FormatTensor(
    const Tensor& t,
    bool all_contiguous,
    bool all_same_dtype,
    MemoryFormat memory_format,
    ScalarType c_type,
    bool copy = false) {
  Tensor o;
  bool view_like = false;
  if (all_same_dtype || t.scalar_type() == c_type) {
    if (all_contiguous || t.is_contiguous(memory_format)) {
      o = FormatContiguous(t.alias(), memory_format);
      view_like = true;
    } else {
      o = t.contiguous(memory_format);
    }
  } else {
    const auto opt = t.options().dtype(c_type).memory_format(memory_format);
    o = at::empty(t.sizes(), opt);
    if (copy) {
      o.copy_(t);
    }
  }

  return std::make_pair(std::move(o), view_like);
}

} // anonymous namespace

TORCH_IMPL_FUNC(CatOut)
(const ITensorListRef& tensors,
 int64_t dim,
 int64_t valid,
 bool all_contiguous,
 bool all_same_dtype,
 bool all_same_sizes_and_stride,
 MemoryFormat memory_format,
 const Tensor& out) {
  (void)valid;
  (void)all_same_sizes_and_stride;

  if (out.numel() == 0) {
    return;
  }

  auto materialized = tensors.materialize();
  // check the complex dtype
  bool has_complex = false;
  for (const Tensor& t : tensors) {
    if (!native::cat_should_skip_tensor(t) && t.numel() > 0) {
      if (at::isComplexType(t.scalar_type())) {
        has_complex = true;
        break;
      }
    }
  }
  if (!has_complex && at::isComplexType(out.scalar_type())) {
    has_complex = true;
  }
  if (has_complex) {
    cat_complex_out_musa(
        tensors,
        dim,
        valid,
        all_contiguous,
        all_same_dtype,
        all_same_sizes_and_stride,
        memory_format,
        out);
    return;
  }

  const size_t n = materialized.size();
  const auto c_type =
      all_same_dtype ? out.scalar_type() : native::result_type(materialized);

  std::vector<Tensor> valid_inputs;
  valid_inputs.reserve(n);
  int elements = 0;

  for (const Tensor& t : tensors) {
    if (!native::cat_should_skip_tensor(t) && t.numel() > 0) {
      auto i_pr = FormatTensor(
          t, all_contiguous, all_same_dtype, memory_format, c_type, true);
      valid_inputs.emplace_back(std::move(i_pr.first));
      ++elements;
    }
  }

  std::vector<muTensor> mu_inputs;
  mu_inputs.reserve(elements);
  for (const Tensor& t : valid_inputs) {
    mu_inputs.emplace_back(CreateMUTensor(t));
  }

  auto o_pr =
      FormatTensor(out, all_contiguous, all_same_dtype, memory_format, c_type);

  auto mu_output = CreateMUTensor(o_pr.first);
  auto& h = at::GetMudnnHandle();
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
      op.Run(h, mu_output, elements, mu_inputs.data()), "ConcatRun");

  if (!o_pr.second) {
    out.copy_(o_pr.first);
  }
}

} // namespace at::musa
