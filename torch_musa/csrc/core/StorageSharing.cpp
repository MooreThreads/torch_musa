#include <ATen/MapAllocator.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include "musa_runtime.h"
#include "torch_musa/csrc/core/MUSAFunctions.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MusaIPCTypes.h"
#include "torch_musa/csrc/core/StorageSharing.h"

static std::string THMPStorageBytesAsHandleString(PyObject* handle) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  char* buffer;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Py_ssize_t handle_size;
  if (PyBytes_AsStringAndSize(handle, &buffer, &handle_size) == -1) {
    // NOLINTNEXTLINE(bugprone-string-constructor)
    return nullptr;
  }
  // NOLINTNEXTLINE(bugprone-string-constructor)
  THPUtils_assert(handle_size == MUSA_IPC_HANDLE_SIZE, "incorrect handle size");
  return std::string(buffer, handle_size);
}

static PyObject* THMPStorageReleaseIPCCounter(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 2, "tuple of 2 items expected");
  PyObject* _ref_counter = PyTuple_GET_ITEM(args, 0);
  PyObject* _ref_counter_offset = PyTuple_GET_ITEM(args, 1);
  if (!(PyBytes_Check(_ref_counter) &&
        THPUtils_checkLong(_ref_counter_offset))) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_release_ipc_counter in MUSA mode",
        1,
        "(bytes _ref_counter, int _ref_counter_offset)");
    return nullptr;
  }
  std::string ref_counter_handle = PyBytes_AS_STRING(_ref_counter);
  ptrdiff_t ref_counter_offset =
      (ptrdiff_t)THPUtils_unpackLong(_ref_counter_offset);
  // We don't want to break existing code, so resource deletion is best
  // effort basis. Exception expected if producer process terminated
  // before consumer released data.
  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE;
  try {
    auto sptr = at::RefcountedMapAllocator::makeDataPtr(
        ref_counter_handle.c_str(),
        flags,
        sizeof(int64_t) * torch::musa::MUSA_IPC_REF_COUNTER_FILE_SIZE,
        nullptr);
    *(static_cast<int64_t*>(sptr.get()) + ref_counter_offset) -= 1;
  } catch (c10::Error& err) {
    // Already warned inside of producer process
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorageShareMusa(
    PyObject* _self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({"_share_musa_(Storage temp)"});
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(_self, args, kwargs, parsed_args);
  auto self = r.storage(0);
  TORCH_CHECK(
      self.device_type() == at::musa::kMUSA,
      "_share_musa_: only available on MUSA, but now ",
      self.device());
  c10::StorageImpl* storage = self.unsafeGetStorageImpl();

  if (storage->received_cuda()) {
    AT_ERROR(
        "Attempted to send MUSA tensor received from another process; this is not currently supported. Consider cloning before sending.");
  }

  at::DeviceGuard device_guard(storage->device());
  THPObjectPtr tuple(PyTuple_New(8));
  THPObjectPtr device(THPUtils_packInt32(storage->device().index()));
  THPObjectPtr _handle(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr size_bytes(THPUtils_packUInt64(storage->nbytes()));
  THPObjectPtr _offset_bytes(THPUtils_packInt32(0));
  THPObjectPtr _ref_counter(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr _ref_counter_offset(THPUtils_packInt32(0));
  THPObjectPtr _event_handle(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr _event_sync_required(Py_None);
  Py_INCREF(Py_None);
  if (storage->data<uint8_t>()) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t base_size;
    void* base_ptr = c10::musa::MUSACachingAllocator::GetBaseAllocation(
        storage->data<uint8_t>(), &base_size);
    ptrdiff_t offset_bytes = (char*)storage->data<uint8_t>() - (char*)base_ptr;

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    musaIpcMemHandle_t handle;
    C10_MUSA_CHECK(musaIpcGetMemHandle(&handle, base_ptr));

    _handle = PyBytes_FromStringAndSize((char*)&handle, MUSA_IPC_HANDLE_SIZE);
    _offset_bytes = PyLong_FromSsize_t((Py_ssize_t)offset_bytes);

    // Put Storage Data behind new ref counting context
    // See Note [MUSA IPC Refcounting implementation explained]
    at::DataPtr sent_data_ptr = torch::musa::GetNewRefCountedSentData(
        storage->data(), storage->device());
    auto old_data_ptr = storage->set_data_ptr(std::move(sent_data_ptr));
    auto sent_data = static_cast<torch::musa::MusaIPCSentData*>(
        storage->data_ptr().get_context());
    sent_data->set_original_ptr(std::move(old_data_ptr));
    _ref_counter = PyBytes_FromString((sent_data->handle()).c_str());
    _ref_counter_offset = THPUtils_packInt64(sent_data->offset());

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    musaIpcEventHandle_t ipc_event_handle;

    if (sent_data->event_sync_required_) {
      C10_MUSA_CHECK(
          musaIpcGetEventHandle(&ipc_event_handle, sent_data->event_));
    }

    _event_handle = PyBytes_FromStringAndSize(
        (char*)&ipc_event_handle, MUSA_IPC_HANDLE_SIZE);
    _event_sync_required = PyBool_FromLong(sent_data->event_sync_required_);
  }

  if (!tuple || !device || !_handle || !size_bytes || !_offset_bytes ||
      !_event_handle) {
    return nullptr;
  }
  PyTuple_SET_ITEM(tuple.get(), 0, device.release());
  // musaIpcMemHandle_t(of basePtr)
  PyTuple_SET_ITEM(tuple.get(), 1, _handle.release());
  // Size(in bytes) of the real storage, note this is not the size of basePtr
  // memory block.
  PyTuple_SET_ITEM(tuple.get(), 2, size_bytes.release());
  // Offset(in bytes) of the real storage in the basePtr memory block.
  // NB: this offset MUST be in bytes instead of numel, since we use
  // (storage_handle, offset)
  //     as key in shared_cache(multiprocessing/reduction.py).
  //     Offset in numel cannot uniquely represent a storage.
  PyTuple_SET_ITEM(tuple.get(), 3, _offset_bytes.release());
  PyTuple_SET_ITEM(tuple.get(), 4, _ref_counter.release());
  PyTuple_SET_ITEM(tuple.get(), 5, _ref_counter_offset.release());
  PyTuple_SET_ITEM(tuple.get(), 6, _event_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 7, _event_sync_required.release());
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPStorageNewSharedMusa(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 8, "tuple of 8 items expected");
  PyObject* _device = PyTuple_GET_ITEM(args, 0);
  PyObject* _handle = PyTuple_GET_ITEM(args, 1);
  PyObject* _size_bytes = PyTuple_GET_ITEM(args, 2);
  PyObject* _offset_bytes = PyTuple_GET_ITEM(args, 3);
  PyObject* _ref_counter = PyTuple_GET_ITEM(args, 4);
  PyObject* _ref_counter_offset = PyTuple_GET_ITEM(args, 5);
  PyObject* _event_handle = PyTuple_GET_ITEM(args, 6);
  PyObject* _event_sync_required = PyTuple_GET_ITEM(args, 7);
  if (!(THPUtils_checkLong(_device) && THPUtils_checkLong(_size_bytes) &&
        PyBytes_Check(_handle) && PyBytes_Check(_ref_counter) &&
        PyBytes_Check(_event_handle) && THPUtils_checkLong(_offset_bytes) &&
        THPUtils_checkLong(_ref_counter_offset) &&
        PyBool_Check(_event_sync_required))) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_new_shared in MUSA mode",
        1,
        "(int device, bytes handle, int storage_size_bytes, int storage_offset_bytes, bytes _ref_counter, int _ref_counter_offset, bytes event_handle, bool event_sync_required)");
    return nullptr;
  }

  size_t storage_size =
      (size_t)THPUtils_unpackLong(_size_bytes) / sizeof(uint8_t);
  ptrdiff_t storage_offset_bytes =
      (ptrdiff_t)THPUtils_unpackLong(_offset_bytes);

  int64_t device = THPUtils_unpackLong(_device);
  at::musa::MUSAGuard device_guard(device);

  if (PyObject_IsTrue(_event_sync_required)) {
    // Ensure that producer prepared all tensor's data
    std::string s_ipc_event_handle =
        THMPStorageBytesAsHandleString(_event_handle);
    auto ipc_event_handle = reinterpret_cast<const musaIpcEventHandle_t*>(
        s_ipc_event_handle.c_str());
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    musaEvent_t event;
    musaIpcOpenEventHandle(&event, *ipc_event_handle);
    C10_MUSA_CHECK(
        musaStreamWaitEvent(c10::musa::getCurrentMUSAStream(device), event, 0));
  }

  std::string s_handle = THMPStorageBytesAsHandleString(_handle);
  std::shared_ptr<void> basePtr =
      c10::musa::MUSACachingAllocator::GetIpcDevPtr(s_handle);

  // Offset the basePtr to reconstruct the real storage
  // devPtr = basePtr + storage_offset
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  void* devPtr = basePtr.get();
  devPtr = (char*)devPtr + storage_offset_bytes;

  std::string ref_counter_handle = PyBytes_AS_STRING(_ref_counter);
  ptrdiff_t ref_counter_offset =
      (ptrdiff_t)THPUtils_unpackLong(_ref_counter_offset);

  struct IpcDeleterContext {
    std::string ref_counter_handle;
    ptrdiff_t ref_counter_offset;
    int64_t device;
    torch::musa::MusaIPCReceivedData received_data;
  };

  auto ctx = std::make_unique<IpcDeleterContext>();
  ctx->ref_counter_handle = std::move(ref_counter_handle);
  ctx->ref_counter_offset = ref_counter_offset;
  ctx->device = device;
  ctx->received_data.shared_ptr_ = std::move(basePtr);

  auto cur_device = c10::musa::current_device();
  c10::DataPtr data_ptr(
      devPtr,
      ctx.release(),
      +[](void* ctx_) {
        std::unique_ptr<IpcDeleterContext> ctx(
            static_cast<IpcDeleterContext*>(ctx_));
        ctx->received_data.shared_ptr_.reset();

        // Sync default stream to make sure all operations related to the
        // storage is finished (otherwise another process may reuse memory and
        // corrupt data)

        // Ideally all shared memory reference counting could be replaced by
        // sending untriggered MUSA event from the producer to consumer and
        // using this event as the criteria of memory release. However, MUSA
        // (atm 10.1) does not support the creation of untriggered events and
        // performance impact of having thousands of shared events is unknown.

        // TODO: Instead of musaStreamSynchronize it is possible to add Stream
        // Callback and release counter inside of it (need to check performance
        // impact)
        c10::musa::stream_synchronize(
            c10::musa::getCurrentMUSAStream(ctx->device));

        // We don't want to break existing code, so resource deletion is best
        // effort basis. Exception expected if producer process terminated
        // before consumer released data.
        int flags =
            at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE;
        try {
          auto sptr = at::RefcountedMapAllocator::makeDataPtr(
              ctx->ref_counter_handle.c_str(),
              flags,
              sizeof(int64_t) * torch::musa::MUSA_IPC_REF_COUNTER_FILE_SIZE,
              nullptr);
          *(static_cast<int64_t*>(sptr.get()) + ctx->ref_counter_offset) -= 1;
        } catch (c10::Error& err) {
          // Already warned inside of producer process
        }
      },
      at::Device(at::musa::kMUSA, cur_device));

  auto base = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      storage_size,
      std::move(data_ptr),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  base->set_resizable(false);
  base->set_received_cuda(true);

  return THPStorage_New(std::move(base));
  END_HANDLE_TH_ERRORS
}

static PyMethodDef THMPStorageSharingMethods[] = {
    {"_share_musa_",
     castPyCFunctionWithKeywords(THMPStorageShareMusa),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_new_shared_musa", THMPStorageNewSharedMusa, METH_VARARGS, nullptr},
    {"_release_ipc_counter_musa",
     THMPStorageReleaseIPCCounter,
     METH_VARARGS,
     nullptr},
    {nullptr}};

PyMethodDef* GetStorageSharingMethods() {
  return THMPStorageSharingMethods;
}
