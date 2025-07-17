#include "torch_musa/csrc/core/Device.h"

#include <pybind11/embed.h>

#include "torch_musa/csrc/core/MUSAFunctions.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace py = pybind11;

namespace c10::musa {

DeviceIndex current_device() {
  DeviceIndex cur_device = -1;
  C10_MUSA_CHECK(GetDevice(&cur_device));
  return cur_device;
}

void set_device(DeviceIndex device) {
  C10_MUSA_CHECK(SetDevice(device));
}

void Synchronize() {
  device_synchronize();
}

void init_mem_get_func(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto musart = m.def_submodule("_musart", "libmusart.so bindings");
  musart.def("musaMemGetInfo", [](int device) -> std::pair<size_t, size_t> {
    c10::musa::MUSAGuard guard(device);
    size_t device_free = 0;
    size_t device_total = 0;
    C10_MUSA_CHECK(musaMemGetInfo(&device_free, &device_total));
    return {device_free, device_total};
  });
}

} // namespace c10::musa
