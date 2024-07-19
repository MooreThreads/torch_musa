#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>

#include <mudnn.h>
#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/musa/MUSAAtomic.muh"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/ops/musa/EmbeddingBackwardKernel.muh"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace native {

namespace {

template <typename scalar_t, typename index_t>
__global__ void EmbeddingDenseBwdAtomicKernel(
    scalar_t* grad_weight,
    const scalar_t* grad,
    const index_t* idx,
    const int num_indices,
    const int tbl_h,
    const int tbl_w,
    const int padding_idx) {
  int ox = threadIdx.x;
  int oy = blockIdx.x + threadIdx.y * gridDim.x;

  for (; oy < num_indices; oy += blockDim.y * gridDim.x) {
    // use a batch of threads with same theadIdx.x
    // to process one row of grad_weight
    index_t id = idx[oy];
    scalar_t* grad_weight_row = grad_weight + id * tbl_w;
    const scalar_t* grad_row = grad + oy * tbl_w;
    bool valid_id = (id >= 0 && id < tbl_h && id != padding_idx);
    for (int i = ox * 2; i < tbl_w; i += (blockDim.x * 2)) {
      if (valid_id) {
        at::musa::gpuAtomicAdd(grad_weight_row + i, grad_row[i]);
      }
    }

    for (int i = ox * 2 + 1; i < tbl_w; i += (blockDim.x * 2)) {
      if (valid_id) {
        at::musa::gpuAtomicAdd(grad_weight_row + i, grad_row[i]);
      }
    }
  }
}

Tensor EmbeddingDenseBwdMUSA(
    const Tensor& grad_output,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  TORCH_CHECK(
      grad_output.device() == indices.device() &&
          grad_output.device().type() == kMUSA,
      "Excepted device of grad_output and indices of embedding_dense_backward "
      "both on MUSA, but now grad_output on ",
      grad_output.device(),
      " indices on ",
      indices.device());
  TORCH_CHECK(
      !scale_grad_by_freq,
      "scale grad by the frequency of the words in the mini-batch is not "
      "supported yet");

  auto contiguous_indices = indices.contiguous();
  auto num_indices = indices.numel();
  auto contiguous_grad_output = grad_output.contiguous();
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  // be careful for setting this value, there may be
  // precision and efficiency drops when value gets larger.
  if (num_indices <= 3072 && !scale_grad_by_freq) {
    Tensor grad_weight = at::zeros(
        {num_weights, grad_output.size(-1)},
        grad_output.options().memory_format(at::MemoryFormat::Contiguous));

    int tbl_h = grad_weight.size(0);
    int tbl_w = grad_weight.size(1);
    const int wrap_size = at::musa::warp_size();
    dim3 block(wrap_size, 8, 1);
    dim3 grid(at::ceil_div(tbl_w, wrap_size), 1, 1);

#define DISPATCH_ATOMIC_KERNEL(SCALAR_TYPE)                     \
  AT_DISPATCH_INDEX_TYPES(                                      \
      indices.scalar_type(), "EmbeddingDenseBwdMUSA", [&]() {   \
        using scalar_t = SCALAR_TYPE;                           \
        EmbeddingDenseBwdAtomicKernel<scalar_t, index_t>        \
            <<<grid, block, 0, stream>>>(                       \
                static_cast<scalar_t*>(grad_weight.data_ptr()), \
                static_cast<const scalar_t*>(                   \
                    contiguous_grad_output.data_ptr()),         \
                contiguous_indices.data_ptr<index_t>(),         \
                static_cast<int>(num_indices),                  \
                static_cast<int>(tbl_h),                        \
                static_cast<int>(tbl_w),                        \
                static_cast<int>(padding_idx));                 \
        C10_MUSA_KERNEL_LAUNCH_CHECK();                         \
      });

    const auto& the_type = contiguous_grad_output.scalar_type();
    switch (the_type) {
      case at::ScalarType::Float:
        DISPATCH_ATOMIC_KERNEL(
            c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Float>);
        break;
      case at::ScalarType::Double:
        DISPATCH_ATOMIC_KERNEL(
            c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Double>);
        break;
      case at::ScalarType::Half:
        // slight performance imporvement compared to ScalarType::Half
        DISPATCH_ATOMIC_KERNEL(float16_t);
        break;
      case at::ScalarType::BFloat16:
        DISPATCH_ATOMIC_KERNEL(bfloat16_t);
        break;
      default:
        AT_ERROR("EmbeddingDenseBwdMUSA not support ", toString(the_type));
    }
#undef DISPATCH_ATOMIC_KERNEL

    return grad_weight;
  }

  auto sorted_indices =
      at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto orig_indices = at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  {
    int64_t numel = contiguous_indices.numel();
    at::musa::muHandle& h = GetMudnnHandle();
    auto indices_ = at::musa::CreateMUTensor(contiguous_indices);
    indices_.SetNdInfo({numel});
    auto orig_indices_ = at::musa::CreateMUTensor(orig_indices);
    orig_indices_.SetNdInfo({numel});
    auto sorted_indices_ = at::musa::CreateMUTensor(sorted_indices);
    sorted_indices_.SetNdInfo({numel});
    ::musa::dnn::Sort op;
    op.SetDim(0);
    op.SetDescending(false);
    op.SetStable(true);
    CHECK_MUDNN_STATUS(
        op.Run(
            h,
            sorted_indices_,
            orig_indices_,
            indices_,
            at::musa::InternalMemAlloc),
        "SortRun");
  }

  return EmbeddingBackwardMUSAKernel(
      contiguous_grad_output,
      orig_indices,
      sorted_indices,
      num_weights,
      padding_idx);
}
} // namespace

REGISTER_MUSA_DISPATCH(embedding_dense_backward_stub, &EmbeddingDenseBwdMUSA);

} // namespace native
} // namespace at
