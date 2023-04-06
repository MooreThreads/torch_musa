#include "torch_musa/csrc/core/Module.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Device.h"

namespace torch_musa {

static void initCommMethods(py::module& m) {
  // Device Management
  m.def("_musa_getDeviceCount", []() { return torch_musa::device_count(); });
  m.def("_musa_getDevice", []() { return torch_musa::current_device(); });
  m.def("_musa_setDevice", [](int device) { torch_musa::set_device(device); });
  // Synchronize musa device.
  m.def("_musa_synchronize", []() { at::native::musa::Synchronize(); });
}
} // namespace torch_musa

void initMUSAModule(PyObject* m) {
  auto t = py::handle(m).cast<py::module>();
  torch_musa::initCommMethods(t);
}
