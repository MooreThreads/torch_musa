#include <ATen/Utils.h>
#include <ATen/autocast_mode.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/invalid_arguments.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>

#include "torch_musa/csrc/amp/autocast_mode.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace autocast {
namespace musa {

static PyObject* GetAutocastMusaDtype(
    PyObject* /* unused */,
    PyObject* /* unused */) {
  HANDLE_TH_ERRORS
  at::ScalarType current_dtype = at::autocast::musa::get_autocast_musa_dtype();
  auto dtype = (PyObject*)torch::getTHPDtype(current_dtype);
  Py_INCREF(dtype);
  return dtype;
  END_HANDLE_TH_ERRORS
}

static PyObject* SetAutocastMusaDtype(PyObject* /* unused */, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!THPDtype_Check(arg)) {
    throw c10::TypeError(
        "dtype must be a torch.dtype (got %s)", Py_TYPE(arg)->tp_name);
  }
  at::ScalarType target_type = reinterpret_cast<THPDtype*>(arg)->scalar_type;
  at::autocast::musa::set_autocast_musa_dtype(target_type);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* SetAutocastMusaEnabled(PyObject* /* unused */, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw c10::TypeError(
        "enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  at::autocast::musa::set_autocast_musa_enabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* IsAutocastMusaEnabled(
    PyObject* /* unused */,
    PyObject* /* unused */) {
  HANDLE_TH_ERRORS
  if (at::autocast::musa::is_autocast_musa_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// autocast methods on torch_musa._MUSAC
static PyMethodDef autocast_methods[] = { // NOLINT
    {"_set_autocast_musa_dtype", SetAutocastMusaDtype, METH_O, nullptr},
    {"_get_autocast_musa_dtype", GetAutocastMusaDtype, METH_NOARGS, nullptr},
    {"_set_autocast_musa_enabled", SetAutocastMusaEnabled, METH_O, nullptr},
    {"_is_autocast_musa_enabled", IsAutocastMusaEnabled, METH_NOARGS, nullptr},
    {nullptr}};

PyMethodDef* GetAutocastMethods() {
  return autocast_methods;
}

} // namespace musa
} // namespace autocast
} // namespace at
