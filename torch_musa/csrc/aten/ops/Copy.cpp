#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>
#pragma GCC diagnostic pop

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

#include <memory>
#include <unordered_set>

namespace at {
namespace native {

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

void mtgpu_impl_copy_d2d(const Tensor& tensor_self, const Tensor& tensor_src) {
  // when tensor_src is empty , we just return
  if (tensor_src.dim() != 0 && tensor_src.numel() == 0) {
    return;
  }
  muHandle h;
  ::musa::dnn::Permute op;
  auto dst_offset = tensor_self.storage_offset();
  auto src_offset = tensor_src.storage_offset();
  auto out_ = CreateMUTensor(tensor_self, true);
  auto in_ = CreateMUTensor(tensor_src, true);
  if (dst_offset || src_offset) {
    CHECK_MUDNN_STATUS(
        op.SetSrcOffset(static_cast<int>(src_offset)), "SetSrcOffset");
    CHECK_MUDNN_STATUS(
        op.SetDstOffset(static_cast<int>(dst_offset)), "SetDstOffset");
  }
  CHECK_MUDNN_STATUS(op.Run(h, out_, in_), "Run");
}

void mtgpu_impl_datacast(const Tensor& tensor_self, const Tensor& tensor_src) {
  // we just copy data there when cast is bool->unint8 and unint8->bool
  if ((tensor_src.dtype() == ScalarType::Bool &&
       tensor_self.dtype() == ScalarType::Byte) ||
      (tensor_src.dtype() == ScalarType::Byte &&
       tensor_self.dtype() == ScalarType::Bool)) {
    mtgpu_impl_copy_d2d(tensor_self, tensor_src);
    return;
  }
  // TODO(caizhi): Since MTDNN does not support Double Cast, it needs to be
  // implemented on the CPU here. This will be improved when adapting MUDNN.
  if (tensor_src.dtype() == ScalarType::Double ||
      tensor_self.dtype() == ScalarType::Double) {
    const std::unordered_set<ScalarType> double_cast_set = {
        ScalarType::Float, ScalarType::Int, ScalarType::Long};
    TORCH_CHECK(
        double_cast_set.find(tensor_src.dtype().toScalarType()) !=
                double_cast_set.end() ||
            double_cast_set.find(tensor_self.dtype().toScalarType()) !=
                double_cast_set.end(),
        "Data type cast from ",
        tensor_src.dtype(),
        " to ",
        tensor_self.dtype(),
        " is not supported on MTGPU now!");
    auto cpu_src = tensor_src.to("cpu");
    auto cpu_result =
        cpu_src.to(tensor_src.options().dtype(tensor_self.dtype()));
    tensor_self.copy_(cpu_result);
    return;
  }

  muHandle h;
  ::musa::dnn::Unary op;

  Tensor src_contig = MusaContiguous(tensor_src);
  Tensor dst_contig = MusaContiguous(tensor_self);

  auto in_ = CreateMUTensor(src_contig);
  auto out_ = CreateMUTensor(dst_contig);

  // data cast float2bool
  if (tensor_src.dtype() == ScalarType::Float &&
      tensor_self.dtype() == ScalarType::Bool) {
    out_.SetType(muTensor::Type::INT8);
  }
  CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::CAST), "SetMode");
  CHECK_MUDNN_STATUS(op.Run(h, out_, in_), "Run");
}

// Note: both cpyfromdevice and cpytodevice will go to this copy_from function!
//       So we should implment both memcpyfrom and memcpyto functions!
inline void mtgpu_impl_copy(
    const Tensor& tensor_self,
    const Tensor& tensor_src,
    Memcpy_type copy_type) {
  muHandle h;
  size_t d_offset = 0;
  void* dev_ptr = nullptr;

  // Since we already check the equivalance of src & dst sizes, so we do not
  // need to check nbytes here.
  const auto capacity = tensor_self.nbytes();
  if (!capacity) {
    return;
  }
  if (copy_type == Memcpy_type::MEMCPY_HOST_TO_DEVICE) { // cpu -> mtgpu
    // Note: tensor.data_ptr() will return the type void*
    d_offset = tensor_self.storage_offset() * tensor_self.itemsize();
    dev_ptr = static_cast<char*>(tensor_self.data_ptr()) - d_offset;
    if (tensor_self.dtype() != tensor_src.dtype()) {
      // Note: when H2D copy, tensor_src and tensor_self have different
      // dtypes, type conversions are performed on the CPU for CPU->GPU copies.
      auto cpu_cast_result = tensor_src.to(tensor_self.dtype());
      auto malloc_res = musaMalloc(&dev_ptr, capacity);
      TORCH_CHECK(
          malloc_res == ::musaError::musaSuccess,
          "Musa Tensor Allocate failed!");
      auto musa_self = CreateMUTensor(tensor_self, true);
      auto result = musa_self.CopyFrom(
          cpu_cast_result.data_ptr(), capacity, musaMemcpyHostToDevice, h);
      TORCH_CHECK(
          result == ::musa::dnn::Status::SUCCESS,
          "Copy(MEMCPY_HOST_TO_DEVICE)");
    } else {
      auto malloc_res = musaMalloc(&dev_ptr, capacity);
      TORCH_CHECK(
          malloc_res == ::musaError::musaSuccess,
          "Musa Tensor Allocate failed!");
      auto musa_self = CreateMUTensor(tensor_self, true);
      auto result = musa_self.CopyFrom(
          tensor_src.data_ptr(), capacity, musaMemcpyHostToDevice, h);
      TORCH_CHECK(
          result == ::musa::dnn::Status::SUCCESS,
          "Copy(MEMCPY_HOST_TO_DEVICE)");
    }
  } else if (copy_type == Memcpy_type::MEMCPY_DEVICE_TO_HOST) { // mtgpu -> cpu
    d_offset = tensor_src.storage_offset() * tensor_src.itemsize();
    dev_ptr = static_cast<char*>(tensor_src.data_ptr()) - d_offset;
    if (tensor_self.dtype() != tensor_src.dtype()) {
      // Note: when D2H copy, tensor_src and tensor_self have different
      // dtypes, type conversions are performed on the CPU for CPU->GPU copies.
      Tensor cpu_tensor = at::empty_like(tensor_self, tensor_src.dtype());
      cpu_tensor.copy_(tensor_src);
      tensor_self.copy_(cpu_tensor);
    } else {
      auto musa_self = CreateMUTensor(tensor_src, true);
      auto result = musa_self.CopyTo(
          tensor_self.data_ptr(), capacity, musaMemcpyDeviceToHost, h);
      TORCH_CHECK(
          result == ::musa::dnn::Status::SUCCESS,
          "Copy(MEMCPY_DEVICE_TO_HOST)");
    }
  } else {
    TORCH_CHECK(false, "Unsupported memcpy type!");
  }
}

} // namespace

Tensor mtgpu_copy_from(
    const Tensor& src,
    const Tensor& self,
    bool non_blocking) {
  (void)non_blocking;
  // For all cases, the source and destination's sizes should be the same.
  TORCH_INTERNAL_ASSERT(self.sizes() == src.sizes());
  // At least one of src and dst should be MTGPU, otherwise it is impossible
  // to fall into this function!
  TORCH_INTERNAL_ASSERT(
      src.device().type() == DeviceType::MTGPU ||
      self.device().type() == DeviceType::MTGPU);

  Memcpy_type copy_type = src.device().type() == DeviceType::CPU
      ? Memcpy_type::MEMCPY_HOST_TO_DEVICE
      : Memcpy_type::MEMCPY_DEVICE_TO_HOST;

  // d2d copy handles all the situations, including pure copy, cast and permute
  if (src.device() == self.device()) {
    // call cast during copy with different type.
    if (src.dtype() != self.dtype()) {
      mtgpu_impl_datacast(self, src);
    } else {
      mtgpu_impl_copy_d2d(self, src);
    }
    return self;
  }

  if (require_copy_backup(src, self)) {
    auto& dst = self;
    Tensor dst_contig;
    Tensor src_contig;

    dst_contig = dst.is_contiguous()
        ? dst
        : at::empty_like(dst, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    auto src_temp = src.itemsize() < dst.itemsize()
        ? src.expand_as(dst)
        : src.to(dst.dtype()).expand_as(dst);
    src_contig = src_temp.is_contiguous()
        ? src_temp
        : src_temp.clone(MemoryFormat::Contiguous);

    // TODO(guandong.lu): expand_as have strided_as op, remember to implement!
    mtgpu_impl_copy(dst_contig, src_contig, copy_type);

    if (!dst_contig.is_same(dst)) {
      TORCH_INTERNAL_ASSERT(dst.device() == dst_contig.device());
      if (dst.device().type() == DeviceType::CPU) {
        dst.copy_(dst_contig);
      } else {
        // call d2d copy to convert intermediate tensor into dst.
        mtgpu_impl_copy_d2d(dst, dst_contig);
      }
    }
  } else {
    mtgpu_impl_copy(self, src, copy_type);
  }
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_copy_from", &mtgpu_copy_from);
}

} // namespace native
} // namespace at
