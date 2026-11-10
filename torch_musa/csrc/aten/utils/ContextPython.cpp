#include <pybind11/pybind11.h>

#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>

#include "torch_musa/csrc/aten/utils/Context.h"

namespace at::musa {

PyObject* THPModuleSetAllowTF32(PyObject* /*unused*/, PyObject* arg) {
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_allow_tf32 expects a bool, but got ",
      THPUtils_typename(arg));
  GlobalContext().SetAllowTF32(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject* THPModuleGetAllowTF32(PyObject* /*unused*/, PyObject* /*unused*/) {
  if (GlobalContext().GetAllowTF32()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModuleSetMublasAllowTF32(PyObject* /*unused*/, PyObject* arg) {
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_mublas_allow_tf32 expects a bool, but got ",
      THPUtils_typename(arg));
  GlobalContext().SetAllowMublasTF32(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject* THPModuleGetMublasAllowTF32(
    PyObject* /*unused*/,
    PyObject* /*unused*/) {
  if (GlobalContext().GetAllowMublasTF32()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyMethodDef ContextMethods[] = { // NOLINT
    {"_get_allow_tf32", THPModuleGetAllowTF32, METH_NOARGS, nullptr},
    {"_set_allow_tf32", THPModuleSetAllowTF32, METH_O, nullptr},
    {"_get_mublas_allow_tf32",
     THPModuleGetMublasAllowTF32,
     METH_NOARGS,
     nullptr},
    {"_set_mublas_allow_tf32", THPModuleSetMublasAllowTF32, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* GetContextMethods() {
  return ContextMethods;
}

} // namespace at::musa
