#ifndef TORCH_MUSA_CSRC_DISTRIBUTED_REGISTER_H_
#define TORCH_MUSA_CSRC_DISTRIBUTED_REGISTER_H_
#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
void registerProcessGroupMCCL(PyObject* mod);

#endif // TORCH_MUSA_CSRC_DISTRIBUTED_REGISTER_H_
