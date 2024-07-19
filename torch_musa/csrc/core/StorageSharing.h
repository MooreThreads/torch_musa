#ifndef TORCH_MUSA_CSRC_CORE_STORAGESHARING_H_
#define TORCH_MUSA_CSRC_CORE_STORAGESHARING_H_

#include <Python.h>
#include <torch/csrc/Storage.h>

PyMethodDef* GetStorageSharingMethods();

#endif // TORCH_MUSA_CSRC_CORE_STORAGESHARING_H_
