#ifndef TORCH_MUSA_CSRC_CORE_MUSA_MODULE_H_
#define TORCH_MUSA_CSRC_CORE_MUSA_MODULE_H_
#include <pybind11/pybind11.h>

namespace py = pybind11;

void initMUSAModule(PyObject* m);

#endif // TORCH_MUSA_CSRC_CORE_MUSA_MUSAEXCEPTION_H_
