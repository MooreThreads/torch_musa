#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/native/TensorFactories.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>
#pragma GCC diagnostic pop

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace detail {

// function: create a mtgpu empty tensor
Tensor empty_mtgpu(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  (void)layout_opt;
  auto device = device_or_default(device_opt);

  bool pin_memory = pinned_memory_or_default(pin_memory_opt);

  TORCH_CHECK(pin_memory == false, "MTGPU only support not pinned memory");
  TORCH_CHECK(device.type() == at::native::musa::kMUSA, "Device isn't MTGPU!");
  c10::Allocator* allocator;

  allocator = c10::GetAllocator(at::native::musa::kMUSA);

  auto dtype = dtype_or_default(dtype_opt);
  constexpr c10::DispatchKeySet mtgpu_ks(at::native::musa::kMUSAKey);
  return empty_generic(size, allocator, mtgpu_ks, dtype, memory_format_opt);
}

} // namespace detail

namespace native {
namespace musa {

// function: resize tensor to a new size
void resize_bytes_mtgpu(StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(
      storage->resizable(), "Trying to resize storage that is not resizable");

  at::DataPtr new_data;
  if (size_bytes != 0) {
    new_data = storage->allocator()->allocate(size_bytes);
  }
  at::DataPtr old_data = storage->set_data_ptr(std::move(new_data));
  const auto old_capacity = storage->nbytes();
  const auto copy_capacity = std::min(size_bytes, old_capacity);
  storage->set_nbytes(size_bytes);
  if (old_data != nullptr && old_data.get() != nullptr && copy_capacity > 0) {
    // need to modify this place for our mtgpu memory storage
    // TODO(guandong.lu): memoryCopy from device to device
    // memcpy(storage->data(), old_data.get(), copy_capacity);
    TORCH_CHECK(false, "MTGPU currently not support copy from D2D");
  }
}

// function: create a new storage or increase the size of the storage
static inline void maybe_resize_storage_mtgpu(
    TensorImpl* self,
    uint64_t new_size) {
  if (new_size == 0) {
    return;
  }

  const auto new_size_bytes_i =
      (new_size + self->storage_offset()) * self->dtype().itemsize();

  const auto new_size_bytes = static_cast<size_t>(new_size_bytes_i);

  const Storage& storage = self->unsafe_storage();
  if (!storage) {
    auto new_storage = c10::make_intrusive<StorageImpl>(
        StorageImpl::use_byte_size_t(),
        new_size_bytes,
        c10::GetAllocator(kMUSA),
        true);
    self->set_storage_keep_dtype(std::move(new_storage));
  } else if (new_size_bytes > storage.nbytes()) {
    resize_bytes_mtgpu(storage.unsafeGetStorageImpl(), new_size_bytes);
  }
}

inline TensorImpl* resize_impl_mtgpu_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool resize_storage = true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    storage_size = storage_size_for(size, *stride);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  // allocate here:
  if (resize_storage) {
    maybe_resize_storage_mtgpu(self, storage_size);
  }
  return self;
}

Tensor empty_mtgpu(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  return at::detail::empty_mtgpu(
      size,
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt,
      memory_format_opt);
}

Tensor empty_strided_mtgpu(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  check_size_nonnegative(size);
  auto t = at::native::musa::empty_mtgpu(
      {0}, dtype_opt, layout_opt, device_opt, pin_memory_opt, c10::nullopt);
  at::native::musa::resize_impl_mtgpu_(t.unsafeGetTensorImpl(), size, stride);
  return t;
}

const Tensor& resize_mtgpu_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
  auto* self_ = self.unsafeGetTensorImpl();
  resize_impl_mtgpu_(self_, size, /*strides=*/c10::nullopt);
  if (optional_memory_format.has_value()) {
    auto memory_format = optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }
  return self;
}

Tensor& set_mtgpu_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(
      Storage::use_byte_size_t(), 0, c10::GetAllocator(kMUSA), true);
  result.set_(storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

Tensor& set_source_(Tensor& result, Storage source) {
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / result.dtype().itemsize());
  return result.set_(source, 0, new_size, {});
}

Tensor& set_storage_mtgpu_(
    Tensor& result,
    Storage storage,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  checkSetStorage(result, storage, storage_offset, size, stride);

  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  c10::optional<IntArrayRef> stride_opt = stride.data() != nullptr
      ? c10::optional<IntArrayRef>(stride)
      : c10::nullopt;
  at::native::musa::resize_impl_mtgpu_(
      result.unsafeGetTensorImpl(), size, stride_opt);
  return result;
}

Tensor& set_tensor_(Tensor& result, const Tensor& source) {
  if (result.unsafeGetTensorImpl() != source.unsafeGetTensorImpl()) {
    return result.set_(
        source.storage(),
        source.storage_offset(),
        source.sizes(),
        source.strides());
  }
  return result;
}

Tensor Contiguous(const Tensor& self, MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format) && !self.storage_offset() &&
      (self.dim() == 0 || (self.dim() != 0 && self.stride(-1) == 1))) {
    return self;
  }
  TORCH_CHECK(
      memory_format != MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");
  return self.clone(memory_format);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &empty_mtgpu);
  m.impl("empty_strided", &empty_strided_mtgpu);
  m.impl("resize_", &resize_mtgpu_);
  m.impl("set_", &set_mtgpu_);
  m.impl("set_.source_Storage_storage_offset", &set_storage_mtgpu_);
  m.impl("set_.source_Storage", &set_source_);
  m.impl("set_.source_Tensor", &set_tensor_);
}

} // namespace musa
} // namespace native
} // namespace at
