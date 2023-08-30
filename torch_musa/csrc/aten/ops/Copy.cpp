#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAEvent.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"
#include "torch_musa/csrc/core/PeerToPeerAccess.h"

#include <mudnn.h>

#include <memory>
#include <unordered_set>

namespace at {
namespace musa {
namespace {

enum class Memcpy_type {
  MEMCPY_HOST_TO_DEVICE,
  MEMCPY_DEVICE_TO_HOST,
};

bool dense_judger_internal(
    int64_t sizes[],
    int64_t strides[],
    int l,
    int r,
    int left_bound,
    int right_bound) {
  if (l > r) {
    return true;
  } else if (l == r) {
    return (strides[l] == left_bound) && (strides[l] * sizes[l] == right_bound);
  } else if (r - l == 1) {
    return (strides[r] > strides[l]) ? (strides[l] == left_bound) &&
            (strides[r] * sizes[r] == right_bound) &&
            (strides[l] * sizes[l] == strides[r])
                                     : (strides[r] == left_bound) &&
            (strides[l] * sizes[l] == right_bound) &&
            (strides[r] * sizes[r] == strides[l]);
  }
  int pivot_index = l;
  int tmp_l = l;
  int tmp_r = r;
  while (tmp_l < tmp_r) {
    while (tmp_l < tmp_r && strides[tmp_r] >= strides[pivot_index]) {
      --tmp_r;
    }
    if (tmp_l >= tmp_r) {
      break;
    }
    std::swap(strides[pivot_index], strides[tmp_r]);
    std::swap(sizes[pivot_index], sizes[tmp_r]);
    pivot_index = tmp_r;
    ++tmp_l;
    while (tmp_l < tmp_r && strides[tmp_l] < strides[pivot_index]) {
      ++tmp_l;
    }
    if (tmp_l >= tmp_r) {
      break;
    }
    std::swap(strides[pivot_index], strides[tmp_l]);
    std::swap(sizes[pivot_index], sizes[tmp_l]);
    pivot_index = tmp_l;
    --tmp_r;
  }
  return dense_judger_internal(
             sizes,
             strides,
             l,
             pivot_index - 1,
             left_bound,
             strides[pivot_index]) &&
      dense_judger_internal(
             sizes,
             strides,
             pivot_index + 1,
             r,
             strides[pivot_index] * sizes[pivot_index],
             right_bound);
}

// This function will check if the data_ptr of a certain Tensor covers an
// contiguous piece of memory.
// It can find out cases that do not need to do contiguous first.
bool check_memory_compatible(const Tensor& self) {
  auto stride_array = self.strides();
  auto size_array = self.sizes();
  int64_t dim_count = self.dim();
  int64_t maxIndex = 0;
  int64_t numel = self.numel();

  if (dim_count == 1) {
    return (stride_array[0] == 1);
  }

  // Now need to copy the strides and size data for sort, is there any better
  // solution?
  std::unique_ptr<int64_t[]> strides(new int64_t[dim_count]);
  std::unique_ptr<int64_t[]> sizes(new int64_t[dim_count]);
  strides[0] = stride_array[0];
  sizes[0] = size_array[0];

  for (int64_t i = 1; i < dim_count; ++i) {
    maxIndex = stride_array[i] > stride_array[maxIndex] ? i : maxIndex;
    strides[i] = stride_array[i];
    sizes[i] = size_array[i];
  }
  return (strides[maxIndex] * sizes[maxIndex] == numel) &&
      dense_judger_internal(
             sizes.get(), strides.get(), 0, dim_count - 1, 1, numel);
}

bool require_copy_backup(const Tensor& src, const Tensor& self) {
  if (src.device().type() == self.device().type()) {
    // We do not handle copy from device to device here.
    return false;
  }
  bool type_check = (src.dtype() == self.dtype());
  bool strides_check = (src.strides() == self.strides());

  if (type_check) {
    if (src.is_contiguous() && self.is_contiguous()) {
      return false;
    } else if (
        !src.is_contiguous() && !self.is_contiguous() &&
        check_memory_compatible(self) && strides_check) {
      // transpose-like. Since all the memory is in contiguous piece, there is
      // no need to copy to another backup.
      return false;
    }
  }

  return true;
}

void permute_to_contiguous(const Tensor& self, const Tensor& src) {
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Permute op;
  auto contiguous_out = CreateMUTensor(self);
  auto contiguous_in = CreateMUTensor(src);
  CHECK_MUDNN_STATUS(op.Run(h, contiguous_out, contiguous_in), "Run");
}

void mtgpu_impl_copy_d2d(
    const Tensor& tensor_self,
    const Tensor& tensor_src,
    bool non_blocking = false) {
  using namespace c10::musa;
  using namespace at::musa;
  // when tensor_src is empty , we just return
  if (tensor_src.dim() != 0 && tensor_src.numel() == 0) {
    return;
  }

  bool same_type = tensor_self.dtype() == tensor_src.dtype();
  bool same_conj = tensor_self.is_conj() == tensor_src.is_conj();
  bool same_neg = tensor_self.is_neg() == tensor_src.is_neg();
  bool is_contig = tensor_self.is_contiguous() && tensor_src.is_contiguous();
  bool memcpy_eligible = same_type && same_conj && same_neg && is_contig;

  Device dst_device = tensor_self.device();
  Device src_device = tensor_src.device();

  MUSAGuard device_guard(src_device);

  MUSAStream copy_stream = getCurrentMUSAStream(src_device.index());
  if (src_device != dst_device) {
    // This is a cross-device copy on the src current stream and dst current
    // stream. We perform a two-way barrier between both devices' streams
    // before the copy. This ensures that any write-after-write and
    // write-after-read dependencies on the destination side are handled, so
    // that no one is operating on the dst memory when we perform the copy.
    // src waits on dst barrier (src already waits on src)
    MUSAEvent dst_ready;
    device_guard.set_device(dst_device);
    dst_ready.record(getCurrentMUSAStream(dst_device.index()));

    device_guard.set_device(src_device);
    dst_ready.block(copy_stream);
  }

  if (memcpy_eligible) {
    bool needs_MemcpyPeer =
        canDeviceAccessPeer(src_device.index(), dst_device.index());
    void* dst = const_cast<void*>(tensor_self.data_ptr());
    void* src = const_cast<void*>(tensor_src.data_ptr());
    size_t size = tensor_src.nbytes();
    if (needs_MemcpyPeer && src_device != dst_device) {
      TORCH_MUSA_CHECK(musaMemcpyPeerAsync(
          dst, dst_device.index(), src, src_device.index(), size, copy_stream));
    } else {
      TORCH_MUSA_CHECK(musaMemcpyAsync(
          dst, src, size, musaMemcpyDeviceToDevice, copy_stream));
    }
  } else {
    TORCH_CHECK(same_type, "Device to device copy is unsupported");
    TORCH_CHECK(same_conj, "Device to device copy is unsupported");
    TORCH_CHECK(same_neg, "Device to device copy is unsupported");
    if (!is_contig) {
      permute_to_contiguous(tensor_self, tensor_src);
      return;
    }
  }

  if (src_device != dst_device) {
    // dst waits on src barrier (dst already waits on dst). We cannot
    // operate on dst's copy until the copy is complete.

    // Still on src_device, record stream event
    MUSAEvent src_ready;
    src_ready.record(copy_stream);

    device_guard.set_device(dst_device);
    src_ready.block(getCurrentMUSAStream(dst_device.index()));
  }

  TORCH_MUSA_CHECK(musaGetLastError());
}

void mtgpu_impl_datacast(const Tensor& tensor_self, const Tensor& tensor_src) {
  c10::musa::MUSAGuard device_guard(tensor_src.device());
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Unary op;

  auto contiguous_in = CreateMUTensor(tensor_src);
  auto contiguous_out = CreateMUTensor(tensor_self);

  CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::CAST), "SetMode");
  CHECK_MUDNN_STATUS(op.Run(h, contiguous_out, contiguous_in), "Run");
}

// Note: both cpyfromdevice and cpytodevice will go to this copy_from function!
//       So we should implment both memcpyfrom and memcpyto functions!
inline void mtgpu_impl_copy(
    const Tensor& tensor_self,
    const Tensor& tensor_src,
    Memcpy_type copy_type,
    bool non_blocking = false) {
  muHandle& h = GetMudnnHandle();

  // Since we already check the equivalance of src & dst sizes, so we do not
  // need to check nbytes here.
  const auto capacity = tensor_self.nbytes();
  if (!capacity) {
    return;
  }
  if (copy_type == Memcpy_type::MEMCPY_HOST_TO_DEVICE) { // cpu -> musa
    // Note: tensor.data_ptr() will return the type void*
    if (tensor_self.dtype() != tensor_src.dtype()) {
      // Note: when H2D copy, tensor_src and tensor_self have different
      // dtypes, type conversions are performed on the CPU for CPU->GPU copies.
      auto cpu_cast_result = tensor_src.to(tensor_self.dtype());
      auto musa_self = CreateMUTensor(tensor_self);
      auto result = musa_self.CopyFrom(
          cpu_cast_result.data_ptr(),
          capacity,
          musaMemcpyHostToDevice,
          h,
          !non_blocking);
      TORCH_CHECK(
          result == ::musa::dnn::Status::SUCCESS,
          "Copy(MEMCPY_HOST_TO_DEVICE)");
    } else {
      auto musa_self = CreateMUTensor(tensor_self);
      auto result = musa_self.CopyFrom(
          tensor_src.data_ptr(),
          capacity,
          musaMemcpyHostToDevice,
          h,
          !non_blocking);
      TORCH_CHECK(
          result == ::musa::dnn::Status::SUCCESS,
          "Copy(MEMCPY_HOST_TO_DEVICE)");
    }
  } else if (copy_type == Memcpy_type::MEMCPY_DEVICE_TO_HOST) { // musa -> cpu
    if (tensor_self.dtype() != tensor_src.dtype()) {
      // Note: when D2H copy, tensor_src and tensor_self have different
      // dtypes, type conversions are performed on the CPU for CPU->GPU copies.
      Tensor cpu_tensor = at::empty_like(tensor_self, tensor_src.dtype());
      cpu_tensor = cpu_tensor.contiguous();
      cpu_tensor.copy_(tensor_src);
      tensor_self.copy_(cpu_tensor);
    } else {
      auto musa_self = CreateMUTensor(tensor_self);
      auto result = musa_self.CopyFrom(
          tensor_src.data_ptr(),
          capacity,
          musaMemcpyDeviceToHost,
          h,
          !non_blocking);
      TORCH_CHECK(
          result == ::musa::dnn::Status::SUCCESS,
          "Copy(MEMCPY_DEVICE_TO_HOST)");
    }
  } else {
    TORCH_CHECK(false, "Unsupported memcpy type!");
  }

  if (tensor_self.is_conj() != tensor_src.is_conj()) {
    tensor_self.conj_physical_();
  }
  if (tensor_self.is_neg() != tensor_src.is_neg()) {
    tensor_self.neg_();
  }
}

} // namespace

static bool maybe_enable_p2p_access(Device dst_device, Device src_device) {
  if (dst_device.is_cpu() || src_device.is_cpu()) {
    return false;
  }
  return at::musa::get_p2p_access(src_device.index(), dst_device.index());
}

Tensor mtgpu_copy_from(
    const Tensor& src,
    const Tensor& self,
    bool non_blocking) {
  // For all cases, the source and destination's sizes should be the same.
  TORCH_INTERNAL_ASSERT(self.sizes() == src.sizes());
  // At least one of src and dst should be MUSA, otherwise it is impossible
  // to fall into this function!
  TORCH_INTERNAL_ASSERT(is_musa(self) || is_musa(src));
  maybe_enable_p2p_access(self.device(), src.device());

  // d2d copy handles all the situations, including pure copy, cast and permute
  if (is_musa(src) && is_musa(self)) {
    // call cast during copy with different type.
    if (src.dtype() == self.dtype()) {
      mtgpu_impl_copy_d2d(self, src);
      return self;
    }

    if (src.device() == self.device()) {
      mtgpu_impl_datacast(self, src);
      return self;
    }

    Tensor dst_contig = src.to(self.dtype());
    mtgpu_impl_copy_d2d(self, dst_contig);

    return self;
  }

  c10::musa::OptionalMUSAGuard device_guard;
  Memcpy_type copy_type;
  if (!is_musa(src) && is_musa(self)) {
    device_guard.set_device(self.device());
    copy_type = Memcpy_type::MEMCPY_HOST_TO_DEVICE;
  } else if (is_musa(src) && !is_musa(self)) {
    device_guard.set_device(src.device());
    copy_type = Memcpy_type::MEMCPY_DEVICE_TO_HOST;
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupport devices in mtGPU copy_()");
  }

  if (require_copy_backup(src, self)) {
    auto& dst = self;
    Tensor dst_contig;
    Tensor src_contig;

    // If non_blocking is true - type conversions are performed on the GPU
    // for CPU-GPU copies, otherwise type conversions are performed on the CPU.
    // Type conversions are performed on the src device for GPU-GPU copies.
    if (is_musa(self) || non_blocking) {
      dst_contig = dst.is_contiguous()
          ? dst
          : at::empty_like(dst, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = src.to(self.scalar_type()).expand_as(dst).contiguous();
    } else {
      bool same_type = self.scalar_type() == src.scalar_type();
      dst_contig = (dst.is_contiguous() && same_type)
          ? dst
          : at::empty_like(
                dst, src.scalar_type(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = src.expand_as(dst).contiguous();
    }
    dst_contig._set_conj(dst.is_conj());
    src_contig._set_conj(self.is_conj());

    dst_contig._set_neg(dst.is_neg());
    src_contig._set_neg(self.is_neg());
    // mtgpu_impl_copy(dst_contig, src_contig, copy_type, non_blocking);
    dst_contig.copy_(src_contig, non_blocking);

    if (!dst_contig.is_same(dst)) {
      TORCH_INTERNAL_ASSERT(dst.device() == dst_contig.device());
      dst.copy_(dst_contig);
    }
  } else {
    mtgpu_impl_copy(self, src, copy_type, non_blocking);
  }
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_copy_from", &mtgpu_copy_from);
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::_copy_from"), TORCH_FN(mtgpu_copy_from));
}

} // namespace musa
} // namespace at
