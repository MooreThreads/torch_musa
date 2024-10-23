#include "torch_musa/csrc/core/Storage.h"

#include <ATen/native/Resize.h>
#include <torch/csrc/utils.h>

namespace at {
namespace musa {
extern void resize_bytes_musa(StorageImpl* storage, size_t size_bytes);

static PyObject* PyMusaStorage_resize_(
    PyObject* /* unused */,
    PyObject* self_and_number_arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      PyTuple_GET_SIZE(self_and_number_arg) == 2, "tuple of 2 item expected");
  PyObject* self = PyTuple_GET_ITEM(self_and_number_arg, 0);
  PyObject* number_arg = PyTuple_GET_ITEM(self_and_number_arg, 1);
  THPStorage_assertNotNull(self);
  const auto& storage = THPStorage_Unpack(self);
  // See Note [Invalid Python Storages]
  auto invalid = storage.data() == nullptr &&
      storage.device_type() != c10::DeviceType::Meta &&
      storage.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid, "Attempted to call resize_() on an invalid python storage.")
  THPUtils_assert(
      THPUtils_checkLong(number_arg),
      "resize_ expects an int, "
      "but got %s",
      THPUtils_typename(number_arg));
  int64_t newsize = THPUtils_unpackLong(number_arg);
  c10::DeviceType device_type = storage.device_type();

  if (device_type == at::kCPU) {
    at::native::resize_bytes_cpu(storage.unsafeGetStorageImpl(), newsize);
  } else if (device_type == at::kPrivateUse1) {
    ptrdiff_t size_bytes_i = newsize;
    TORCH_CHECK(
        !c10::overflows<size_t>(size_bytes_i),
        "Requested storage size (",
        size_bytes_i,
        ") cannot be represented as a size_t");
    const auto size_bytes = static_cast<size_t>(size_bytes_i);
    at::musa::resize_bytes_musa(storage.unsafeGetStorageImpl(), size_bytes);
  } else {
    TORCH_CHECK(
        false,
        "UntypedStorage.resize_: got unexpected device type ",
        device_type);
  }
  Py_INCREF(self);
  return self;
  END_HANDLE_TH_ERRORS
}

static PyMethodDef MusaStorageMethods[] = {
    {"_musa_storage_resize_", PyMusaStorage_resize_, METH_VARARGS, nullptr},
    {nullptr}};

PyMethodDef* GetStorageMethods() {
  return MusaStorageMethods;
}

} // namespace musa
} // namespace at
