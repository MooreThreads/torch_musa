#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/mm.h>
#endif

#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at::musa {

using at::native::get_nested_tensor_impl;
using at::native::get_offset_for_index;
using at::native::get_size_for_index;
using at::native::get_stride_for_index;
using at::native::wrap_buffer;

Tensor BmmNested(const Tensor& self, const Tensor& mat2) {
  const c10::musa::MUSAGuard device_guard(self.device());

  // dispatcher should have guaranteed that at least one is nested
  auto self_impl = self.is_nested() ? get_nested_tensor_impl(self)
                                    : self.unsafeGetTensorImpl();
  auto mat2_impl = mat2.is_nested() ? get_nested_tensor_impl(mat2)
                                    : mat2.unsafeGetTensorImpl();

  TORCH_CHECK(self_impl->dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(mat2_impl->dim() == 3, "batch2 must be a 3D tensor");

  const int64_t ntensors = self_impl->size(0);
  const int64_t ntensors2 = mat2_impl->size(0);
  TORCH_CHECK(
      ntensors == ntensors2,
      "Expected size for the 1st dimension of batch2 tensor to be: ",
      ntensors,
      " but got: ",
      ntensors2,
      ".");

  // create a contiguous output NestedTensor size matrix
  const Tensor& self_sizemat = self.is_nested()
      ? get_nested_tensor_impl(self)->get_nested_sizes()
      : get_nested_tensor_impl(mat2)->get_nested_sizes();

  Tensor out_sizemat = self_sizemat.new_empty(self_sizemat.sizes());
  auto* out_sizemat_ptr = out_sizemat.data_ptr<int64_t>();

  int64_t out_numel = 0;
  for (int64_t i = 0; i < ntensors; i++) {
    const IntArrayRef& self_shape = get_size_for_index(self, i);
    const IntArrayRef& mat2_shape = get_size_for_index(mat2, i);
    const int64_t& self_size0 = self_shape[0];
    const int64_t& self_size1 = self_shape[1];
    const int64_t& mat2_size0 = mat2_shape[0];
    const int64_t& mat2_size1 = mat2_shape[1];
    TORCH_CHECK(
        self_size1 == mat2_size0,
        i,
        "-th nested matrices in batch cannot be multiplied (",
        self_size0,
        "x",
        self_size1,
        " and ",
        mat2_size0,
        "x",
        mat2_size1,
        ")");
    out_sizemat_ptr[0] = self_size0;
    out_sizemat_ptr[1] = mat2_size1;
    out_sizemat_ptr += 2;
    out_numel += self_size0 * mat2_size1;
  }

  const Tensor& self_buffer = self.is_nested()
      ? get_nested_tensor_impl(self)->get_unsafe_storage_as_tensor()
      : self;
  const Tensor& mat2_buffer = mat2.is_nested()
      ? get_nested_tensor_impl(mat2)->get_unsafe_storage_as_tensor()
      : mat2;

  Tensor out_buffer = self_buffer.new_empty(out_numel);
  Tensor output = wrap_buffer(out_buffer, out_sizemat);
  auto* out_impl = get_nested_tensor_impl(output);
  const int64_t* out_offsets_ptr =
      out_impl->get_storage_offsets().const_data_ptr<int64_t>();
  (void)out_offsets_ptr; // unused for now, but kept for parity with CUDA impl

  // Fallback path: for each nested matrix, call mm_out which dispatches to
  // MUSA dense matmul implementation.
  std::vector<Tensor> output_unbind = output.unbind();
  for (int64_t i = 0; i < ntensors; i++) {
    at::mm_out(
        output_unbind[i],
        self_buffer.as_strided(
            get_size_for_index(self, i),
            get_stride_for_index(self, i),
            get_offset_for_index(self, i)),
        mat2_buffer.as_strided(
            get_size_for_index(mat2, i),
            get_stride_for_index(mat2, i),
            get_offset_for_index(mat2, i)));
  }

  return output;
}

} // namespace at::musa
