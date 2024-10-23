#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/native/TensorFactories.h>
#include <c10/core/Allocator.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Allocator.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/PeerToPeerAccess.h"
#include "torch_musa/csrc/utils/musa_lazy_init.h"

#include <mudnn.h>

namespace at {
namespace detail {

// function: create a musa empty tensor
Tensor empty_musa(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  at::musa::lazyInitMUSA();
  if (layout_opt.has_value()) {
    LOG(INFO) << "layout_opt is invalid in empty_musa";
  }
  auto device = device_or_default(device_opt);
  c10::musa::OptionalMUSAGuard guard(device);

  TORCH_CHECK(device.type() == at::musa::kMUSA, "Device isn't MUSA!");
  c10::Allocator* allocator = c10::musa::MUSACachingAllocator::get();

  auto dtype = dtype_or_default(dtype_opt);
  constexpr c10::DispatchKeySet musa_ks(at::musa::kMUSAKey);
  if (size.size() != 4 && size.size() != 5) {
    return empty_generic(
        size, allocator, musa_ks, dtype, at::MemoryFormat::Contiguous);
  }
  return empty_generic(size, allocator, musa_ks, dtype, memory_format_opt);
}

TensorBase empty_musa(IntArrayRef size, const TensorOptions& options) {
  return at::detail::empty_musa(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

Tensor empty_strided_musa(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  at::musa::lazyInitMUSA();
  check_size_nonnegative(size);

  TORCH_CHECK(
      !pin_memory_opt.has_value() || !*pin_memory_opt,
      "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      layout_or_default(layout_opt) == Layout::Strided);

  auto dtype = dtype_or_default(dtype_opt);
  auto device = device_or_default(device_opt);

  TORCH_CHECK(device.type() == at::musa::kMUSA, "Device isn't MUSA!");
  c10::musa::OptionalMUSAGuard guard(device);
  c10::Allocator* allocator = c10::musa::MUSACachingAllocator::get();
  constexpr c10::DispatchKeySet musa_dispatch_key(at::musa::kMUSAKey);
  return at::detail::empty_strided_generic(
      size, stride, allocator, musa_dispatch_key, dtype);
}

Tensor empty_strided_musa(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options) {
  return at::detail::empty_strided_musa(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

} // namespace detail

namespace musa {
// function: resize tensor to a new size
void resize_bytes_musa(StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(
      storage->resizable(), "Trying to resize storage that is not resizable");
  auto allocator = storage->allocator();
  TORCH_CHECK(
      allocator != nullptr, "Trying to resize storage without an allocator");

  auto device = storage->device();
  if (size_bytes == 0) {
    storage->set_data_ptr_noswap(at::DataPtr(nullptr, device));
    storage->set_nbytes(0);
    return;
  }

  c10::musa::MUSAGuard guard(device.index());
  at::DataPtr data = allocator->allocate(size_bytes);
  if (storage->data_ptr()) {
    // Enable p2p access when the memcpy is across devices
    at::musa::lazyInitMUSA();

    // Is there cross device copy scenario ?
    // at::musa::get_p2p_access(device, device.index());
    C10_MUSA_CHECK(musaMemcpyAsync(
        data.get(),
        storage->data(),
        std::min(storage->nbytes(), size_bytes),
        musaMemcpyDeviceToDevice,
        c10::musa::getCurrentMUSAStream()));
  }

  // Destructively overwrite data_ptr
  storage->set_data_ptr_noswap(std::move(data));
  storage->set_nbytes(size_bytes);
}

// function: create a new storage or increase the size of the storage
static inline void maybe_resize_storage_musa(
    TensorImpl* self,
    uint64_t new_size_bytes) {
  const Storage& storage = self->unsafe_storage();
  if (!storage) {
    auto new_storage = c10::make_intrusive<StorageImpl>(
        StorageImpl::use_byte_size_t(),
        new_size_bytes,
        c10::musa::MUSACachingAllocator::get(),
        true);
    self->set_storage_keep_dtype(std::move(new_storage));
  } else if (self->numel() == 0) {
    // Skip resizing if the storage doesn't contain anything.
    return;
  } else if (new_size_bytes > storage.nbytes()) {
    resize_bytes_musa(storage.unsafeGetStorageImpl(), new_size_bytes);
  }
}

inline TensorImpl* resize_impl_musa_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  auto itemsize = self->dtype().itemsize();
  auto storage_offset = self->storage_offset();
  size_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    storage_size = at::detail::computeStorageNbytes(
        size, *stride, itemsize, storage_offset);

  } else {
    self->set_sizes_contiguous(size);
    storage_size = at::detail::computeStorageNbytesContiguous(
        size, itemsize, storage_offset);
  }
  maybe_resize_storage_musa(self, storage_size);

  return self;
}

Tensor EmptyMUSA(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  c10::musa::OptionalMUSAGuard guard(device_opt);
  return at::detail::empty_musa(
      size,
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt,
      memory_format_opt);
}

Tensor EmptyStridedMUSA(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  c10::musa::OptionalMUSAGuard guard(device_opt);
  return at::detail::empty_strided_musa(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

void check_inplace(
    const Tensor& self,
    IntArrayRef sizes,
    const TensorOptions& options) {
  // These checks are needed on those operators that:
  //   1) don't use 'TensorIterator' (e.g. 'addmm' and 'baddbmm')
  //   2) have particular typing rules (e.g. 'cumsum' and 'cumprod')
  // For other operators (e.g. 'add'), 'TensorIterator' already checks
  // these things separately.
  TORCH_CHECK(
      options.dtype() == self.dtype(),
      "Bad in-place call: ",
      "input tensor dtype ",
      self.dtype(),
      " and output tensor dtype ",
      options.dtype(),
      " should match");
  TORCH_CHECK(
      options.device() == self.device(),
      "Bad in-place call: ",
      "input tensor device ",
      self.device(),
      " and output tensor device ",
      options.device(),
      " should match");
  TORCH_CHECK(
      sizes == self.sizes(),
      "Bad in-place call: ",
      "input tensor size ",
      self.sizes(),
      " and output tensor size ",
      sizes,
      " should match");
}

void resize_out(
    const Tensor& out,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options) {
  TORCH_CHECK(
      options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ",
      options.dtype(),
      ", but got ",
      out.dtype(),
      " instead");
  TORCH_CHECK(
      options.device() == out.device(),
      "Expected out tensor to have device ",
      options.device(),
      ", but got ",
      out.device(),
      " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      at::as_strided_(out, sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(
          *options.memory_format_opt());
    }
  }
}

Tensor create_out(
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options) {
  if (strides.empty()) {
    return at::detail::empty_musa(
        sizes,
        optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(),
        options.device_opt(),
        options.pinned_memory_opt(),
        options.memory_format_opt());
  } else {
    // TODO(mt-ai): use memory_format in options
    return EmptyStridedMUSA(
        sizes,
        strides,
        optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(),
        options.device_opt(),
        options.pinned_memory_opt());
  }
}

const Tensor& ResizeMUSA_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return at::native::resize_named_tensor_(self, size, optional_memory_format);
  }
  auto* contiguous_self = self.unsafeGetTensorImpl();
  resize_impl_musa_(contiguous_self, size, /*strides=*/c10::nullopt);
  if (optional_memory_format.has_value()) {
    auto memory_format = optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    contiguous_self->empty_tensor_restride(memory_format);
  }
  return self;
}

Tensor& SetMUSA_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(
      Storage::use_byte_size_t(),
      0,
      c10::musa::MUSACachingAllocator::get(),
      true);
  result.set_(storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

Tensor& SetSource_(Tensor& result, Storage source) {
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / result.dtype().itemsize());
  return result.set_(source, 0, new_size, {});
}

Tensor& SetStorageMUSA_(
    Tensor& result,
    Storage storage,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  at::native::checkSetStorage(result, storage, storage_offset, size, stride);

  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  c10::optional<IntArrayRef> stride_opt = stride.data() != nullptr
      ? c10::optional<IntArrayRef>(stride)
      : c10::nullopt;
  at::musa::resize_impl_musa_(result.unsafeGetTensorImpl(), size, stride_opt);
  return result;
}

Tensor& SetTensor_(Tensor& result, const Tensor& source) {
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
  if (4 == self.dim() && self.is_contiguous() &&
      self.is_contiguous(MemoryFormat::ChannelsLast)) {
    if (self.stride(1) != 1 && self.stride(3) != 1) {
      // Why we still need to check the stride at index1 and index3 ?
      // assume that the current input Tensor has shape (4, 1, 16, 1) and stride
      // (16, 16, 1, 16), in this case, MemoryFormat::Contiguous ==
      // input.suggest_memory_format() always equal to true, which hinders to
      // get the desired Contiguous format result. See
      // https://github.com/pytorch/pytorch/blob/c263bd43e8e8502d4726643bc6fd046f0130ac0e/aten/src/ATen/native/TensorConversions.cpp#L377-L393
      // for more details of Tensor.to()
      Tensor temp = at::empty_like(self, memory_format);
      temp.copy_(self);
      return temp;
    }
    return self.to(memory_format);
  }
  return self.contiguous(memory_format);
}

Tensor ContiguousRef(
    const Tensor& self,
    Tensor& ref,
    MemoryFormat memory_format) {
  ref = self.contiguous(memory_format);
  return ref;
}

namespace {

static at::Tensor& EyeOutImpl(int64_t n, int64_t m, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
  TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);
  result.resize_({n, m});
  result.zero_();

  int64_t sz = std::min<int64_t>(n, m);
  int64_t stride = result.stride(0) + result.stride(1);

  Tensor diag = result.as_strided({sz}, {stride});
  diag.fill_(1);
  return result;
}

} // anonymous namespace

at::Tensor& EyeOut(int64_t n, at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, out, "EyeOut", "out");
  const OptionalDeviceGuard device_guard(device_of(out));
  return EyeOutImpl(n, n, out);
}

at::Tensor& EyeMOut(int64_t n, int64_t m, at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, out, "EyeMOut", "out");
  const OptionalDeviceGuard device_guard(device_of(out));
  return EyeOutImpl(n, m, out);
}

at::Tensor Take(const at::Tensor& self, const at::Tensor& index) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "Take", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "Take", "index");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::take(self, index);
}

at::Tensor& Put_(
    at::Tensor& self,
    const at::Tensor& index,
    const at::Tensor& source,
    bool accumulate) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "Put_", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "Put_", "index");
  c10::impl::check_and_update_common_device(
      common_device, source, "Put_", "source");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::put_(self, index, source, accumulate);
}

} // namespace musa
} // namespace at
