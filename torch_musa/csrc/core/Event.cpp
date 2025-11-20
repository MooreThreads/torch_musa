#include "torch_musa/csrc/core/Event.h"

#include <musa_runtime_api.h>
#include <pybind11/pybind11.h>
#include <structmember.h>

#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/Stream.h"

PyObject* THMPEventClass = nullptr;

static PyObject* THMPEvent_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  unsigned char enable_timing = 0;
  unsigned char blocking = 0;
  unsigned char interprocess = 0;

  constexpr const char* kwlist[] = {
      "enable_timing", "blocking", "interprocess", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|bbb",
          const_cast<char**>(kwlist),
          &enable_timing,
          &blocking,
          &interprocess)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THMPEvent* self = (THMPEvent*)ptr.get();
  unsigned int flags = (blocking ? musaEventBlockingSync : musaEventDefault) |
      (enable_timing ? musaEventDefault : musaEventDisableTiming) |
      (interprocess ? musaEventInterprocess : musaEventDefault);

  new (&self->musa_event) at::musa::MUSAEvent(flags);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_from_ipc_handle(
    PyObject* _type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto type = (PyTypeObject*)_type;

  static torch::PythonArgParser parser({
      "from_ipc_handle(Device device, std::string ipc_handle)",
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  at::Device device = r.device(0);
  std::string handle_string = r.string(1);

  TORCH_CHECK(
      handle_string.size() == sizeof(musaIpcEventHandle_t),
      "musaIpcEventHandle_t expects byte-like object of size ",
      sizeof(musaIpcEventHandle_t),
      ", but got ",
      handle_string.size());
  TORCH_CHECK(
      device.type() == at::musa::kMUSA,
      "Event can only be created on "
      "MUSA devices, but got device type ",
      device.type())

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }
  THMPEvent* self = (THMPEvent*)ptr.get();

  musaIpcEventHandle_t handle;
  std::memcpy(&handle, handle_string.c_str(), handle_string.size());
  new (&self->musa_event) at::musa::MUSAEvent(device.index(), &handle);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THMPEvent_dealloc(THMPEvent* self) {
  {
    pybind11::gil_scoped_release no_gil{};
    self->musa_event.~MUSAEvent();
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THMPEvent_get_musa_event(THMPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->musa_event.event());
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_get_device(THMPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  std::optional<at::Device> device = self->musa_event.device();
  if (!device) {
    Py_RETURN_NONE;
  }
  return THPDevice_New(device.value());
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_record(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS {
    auto self = reinterpret_cast<THMPEvent*>(_self);
    auto stream = reinterpret_cast<THMPStream*>(_stream);
    pybind11::gil_scoped_release no_gil{};
    self->musa_event.record(stream->musa_stream);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_wait(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS {
    auto self = reinterpret_cast<THMPEvent*>(_self);
    auto stream = reinterpret_cast<THMPStream*>(_stream);
    pybind11::gil_scoped_release no_gil{};
    self->musa_event.block(stream->musa_stream);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_query(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = reinterpret_cast<THMPEvent*>(_self);
  return PyBool_FromLong(self->musa_event.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_elapsed_time(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto self = reinterpret_cast<THMPEvent*>(_self);
  auto other = reinterpret_cast<THMPEvent*>(_other);
  return PyFloat_FromDouble(self->musa_event.elapsed_time(other->musa_event));
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    auto self = reinterpret_cast<THMPEvent*>(_self);
    pybind11::gil_scoped_release no_gil{};
    self->musa_event.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THMPEvent_ipc_handle(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = reinterpret_cast<THMPEvent*>(_self);
  musaIpcEventHandle_t handle;
  self->musa_event.ipc_handle(&handle);
  return PyBytes_FromStringAndSize((const char*)&handle, sizeof(handle));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THMPEvent_properties[] = {
    {"device", (getter)THMPEvent_get_device, nullptr, nullptr, nullptr},
    {"musa_event", (getter)THMPEvent_get_musa_event, nullptr, nullptr, nullptr},
    {nullptr}};

static PyMethodDef THMPEvent_methods[] = {
    {(char*)"from_ipc_handle",
     castPyCFunctionWithKeywords(THMPEvent_from_ipc_handle),
     METH_CLASS | METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {(char*)"record", THMPEvent_record, METH_O, nullptr},
    {(char*)"wait", THMPEvent_wait, METH_O, nullptr},
    {(char*)"query", THMPEvent_query, METH_NOARGS, nullptr},
    {(char*)"elapsed_time", THMPEvent_elapsed_time, METH_O, nullptr},
    {(char*)"synchronize", THMPEvent_synchronize, METH_NOARGS, nullptr},
    {(char*)"ipc_handle", THMPEvent_ipc_handle, METH_NOARGS, nullptr},
    {nullptr}};

PyTypeObject THMPEventType = {
    PyVarObject_HEAD_INIT(
        nullptr,
        0) "torch_musa._MUSAC._MusaEventBase", /* tp_name */
    sizeof(THMPEvent), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THMPEvent_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THMPEvent_methods, /* tp_methods */
    nullptr, /* tp_members */
    THMPEvent_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THMPEvent_pynew, /* tp_new */
};

void THMPEvent_init(PyObject* module) {
  TORCH_CHECK(THPEventClass, "THPEvent has not been initialized yet.");
  Py_INCREF(THPEventClass);
  THMPEventType.tp_base = THPEventClass;
  THMPEventClass = (PyObject*)&THMPEventType;
  if (PyType_Ready(&THMPEventType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THMPEventType);
  if (PyModule_AddObject(module, "_MusaEventBase", (PyObject*)&THMPEventType) <
      0) {
    throw python_error();
  }
}
