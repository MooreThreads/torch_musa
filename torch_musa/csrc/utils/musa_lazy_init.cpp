#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

#include "torch_musa/csrc/utils/musa_lazy_init.h"

namespace torch {
namespace utils {
namespace musa {

bool is_initialized = false;

} // namespace musa

void musa_lazy_init() {
  pybind11::gil_scoped_acquire g;
  // Protected by the GIL.  We don't use call_once because under ASAN it
  // has a buggy implementation that deadlocks if an instance throws an
  // exception.  In any case, call_once isn't necessary, because we
  // have taken a lock.
  if (musa::is_initialized) {
    return;
  }

  auto module = THPObjectPtr(PyImport_ImportModule("torch_musa"));
  if (!module) {
    throw python_error();
  }

  auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
  if (!res) {
    throw python_error();
  }

  musa::is_initialized = true;
}

void set_requires_musa_init(bool value) {
  musa::is_initialized = !value;
}

} // namespace utils
} // namespace torch
