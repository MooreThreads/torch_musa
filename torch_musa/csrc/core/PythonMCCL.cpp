#include "torch_musa/csrc/core/PythonMCCL.h"

#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>

#include "torch_musa/csrc/core/MCCL.h"

PyObject* THMPModule_mccl_version(PyObject* self, PyObject* args) {
  using torch::musa::mccl::version;
  return PyLong_FromUnsignedLongLong(version());
}

PyObject* THMPModule_mccl_version_suffix(PyObject* self, PyObject* args) {
  using torch::musa::mccl::version_suffix;
  HANDLE_TH_ERRORS
  return PyBytes_FromString(version_suffix());
  END_HANDLE_TH_ERRORS
}
