#ifndef TORCH_MUSA_CSRC_CORE_PYTHONMCCL_H_
#define TORCH_MUSA_CSRC_CORE_PYTHONMCCL_H_

#include <torch/csrc/python_headers.h>

PyObject* THMPModule_mccl_version(PyObject* self, PyObject* args);
PyObject* THMPModule_mccl_version_suffix(PyObject* self, PyObject* args);

#endif // TORCH_MUSA_CSRC_CORE_PYTHONMCCL_H_
