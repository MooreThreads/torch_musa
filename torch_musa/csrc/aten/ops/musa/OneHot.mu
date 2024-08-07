#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

#include <ATen/ops/one_hot_native.h>

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/ops/OneHot.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace native {

template <typename SrcDtype, typename DstDtype>
__global__ void OneHot(
    DstDtype* out,
    const SrcDtype* in,
    const int depth,
    const int n,
    const int suffix_size,
    DstDtype on_value,
    DstDtype off_value,
    const at::musa::FastDivmod fdm_ds,
    const at::musa::FastDivmod fdm_su) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride = blockDim.x * gridDim.x;

  for (int index = tid; index < n; index += stride) {
    uint32_t prefix_idx, prefix_off, depth_idx, suffix_idx;
    fdm_ds(prefix_idx, prefix_off, index);
    fdm_su(depth_idx, suffix_idx, prefix_off);

    int in_idx = prefix_idx * suffix_size + suffix_idx;

    // TODO(@mt-ai/mt-sw-compute): Currently we do not support negative index
    bool valid_index = in[in_idx] >= 0 && in[in_idx] < depth;

    out[index] = (valid_index && in[in_idx] == SrcDtype(depth_idx)) ? on_value
                                                                    : off_value;
  }
}

namespace {
template <typename SrcDtype, typename DstDtype>
void LaunchOneHotKernel(
    Tensor& o,
    const Tensor& i,
    const int depth,
    const int n,
    const int suffix_size,
    DstDtype on_value,
    DstDtype off_value) {
  at::musa::muHandle& h = GetMudnnHandle();
  auto stream = c10::musa::getCurrentMUSAStream();
  at::musa::FastDivmod fdm_ds(depth * suffix_size);
  at::musa::FastDivmod fdm_su(suffix_size);

  musaDeviceProp device_prop;
  int device_id = h.GetDeviceId();
  TORCH_CHECK(
      musaSuccess == musaGetDeviceProperties(&device_prop, device_id),
      "musaGetDeviceProperties error");
  const int mp_num = device_prop.multiProcessorCount;
  const uint32_t threads = 1024;
  const uint32_t blocks = std::min(at::musa::ceil_div(n, 1024), mp_num);

  OneHot<SrcDtype, DstDtype><<<blocks, threads, 0, stream>>>(
      static_cast<DstDtype*>(o.data_ptr()),
      static_cast<SrcDtype*>(i.data_ptr()),
      depth,
      n,
      suffix_size,
      on_value,
      off_value,
      fdm_ds,
      fdm_su);
}

template <typename DstDtype>
void DispatchOneHotWithDstDtype(
    Tensor& o,
    const Tensor& i,
    const int depth,
    const int n,
    const int suffix_size,
    DstDtype on_value,
    DstDtype off_value) {
  switch (i.scalar_type()) {
    case at::ScalarType::Int:
      LaunchOneHotKernel<int32_t, DstDtype>(
          o, i, depth, n, suffix_size, on_value, off_value);
      break;
    case at::ScalarType::Float:
      LaunchOneHotKernel<float, DstDtype>(
          o, i, depth, n, suffix_size, on_value, off_value);
      break;
    case at::ScalarType::Long:
      LaunchOneHotKernel<int64_t, DstDtype>(
          o, i, depth, n, suffix_size, on_value, off_value);
      break;
    default:
      TORCH_CHECK(false, "Unsupported input data type: ", i.scalar_type());
  }
}

bool CheckParams(
    const Tensor& o,
    const Tensor& i,
    int& axis,
    int& prefix_size,
    int& suffix_size,
    const int depth) {
  TORCH_CHECK(depth > 0, "Depth should be positive, but got ", depth);
  int rank = o.dim();
  TORCH_CHECK(
      rank == i.dim() + 1,
      "Output tensor's rank shouble be ",
      i.dim() + 1,
      ", but got ",
      rank);
  axis = rank - 1;

  for (int idx = 0; idx < axis; ++idx) {
    TORCH_CHECK(
        o.sizes()[idx] == i.sizes()[idx],
        "Tensor shape dismatch. Expected output dim is ",
        i.sizes()[idx],
        ", but got ",
        o.sizes()[idx]);
    prefix_size *= i.sizes()[idx];
  }
  TORCH_CHECK(
      o.sizes()[axis] == depth,
      "Tensor shape dismatch. Expected output dim is ",
      depth,
      ", but got ",
      o.sizes()[axis]);
  for (int idx = axis + 1; idx < o.dim(); ++idx) {
    TORCH_CHECK(
        o.sizes()[idx] == i.sizes()[idx - 1],
        "Tensor shape dismatch. Expected output dim is ",
        i.sizes()[idx - 1],
        ", but got ",
        o.sizes()[idx - 1]);
  }
  suffix_size = i.numel() / prefix_size;
  return true;
}

} // namespace

void OneHotRun(Tensor& o, const Tensor& i, const int num_classes) {
  int axis = -1;
  int prefix_size = 1;
  int suffix_size = 1;
  TORCH_CHECK(
      CheckParams(o, i, axis, prefix_size, suffix_size, num_classes),
      "CheckParams Fail");

  int n = o.numel();

  switch (o.scalar_type()) {
    case at::ScalarType::Int:
      return DispatchOneHotWithDstDtype<int32_t>(
          o,
          i,
          num_classes,
          n,
          suffix_size,
          static_cast<int32_t>(1),
          static_cast<int32_t>(0));
    case at::ScalarType::Long:
      return DispatchOneHotWithDstDtype<int64_t>(
          o,
          i,
          num_classes,
          n,
          suffix_size,
          static_cast<int64_t>(1),
          static_cast<int64_t>(0));
    case at::ScalarType::Float:
      return DispatchOneHotWithDstDtype<float>(
          o,
          i,
          num_classes,
          n,
          suffix_size,
          static_cast<float>(1),
          static_cast<float>(0));
    default:
      TORCH_CHECK(false, "Unsupported output data type: ", o.scalar_type());
  }
}

REGISTER_MUSA_DISPATCH(onehot_stub, &OneHotRun);

} // namespace native
} // namespace at
