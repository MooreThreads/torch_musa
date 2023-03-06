#include "torch_musa/csrc/aten/musa/MUSAContext.h"

#include <deque>
#include <mutex>
#include <vector>

#include <ATen/musa/MUSAConfig.h>
#include <c10/util/CallOnce.h>

#include "torch_musa/csrc/core/Allocator.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAException.h"

namespace at {
namespace musa {
namespace {

DeviceIndex num_gpus = -1;
c10::once_flag init_flag;
std::deque<c10::once_flag> device_flags;
std::vector<musaDeviceProp> device_properties;

void initMUSAContextVectors() {
  num_gpus = c10::musa::device_count();
  device_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
}

void initDeviceProperty(DeviceIndex device_index) {
  musaDeviceProp device_prop;
  TORCH_MUSA_CHECK(musaGetDeviceProperties(&device_prop, device_index));
  device_properties[device_index] = device_prop;
}

} // anonymous namespace

/* Device info */
int warp_size() {
  return getCurrentDeviceProperties()->warpSize;
}

musaDeviceProp* getCurrentDeviceProperties() {
  auto device = c10::musa::current_device();
  return getDeviceProperties(device);
}

musaDeviceProp* getDeviceProperties(int device) {
  c10::call_once(init_flag, initMUSAContextVectors);
  if (device == -1)
    device = current_device();
  AT_ASSERT(device >= 0 && device < num_gpus);
  c10::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

bool canDeviceAccessPeer(int device, int peer_device) {
  c10::call_once(init_flag, initMUSAContextVectors);
  if (device == -1)
    device = current_device();
  AT_ASSERT(device >= 0 && device < num_gpus);
  AT_ASSERT(peer_device >= 0 && peer_device < num_gpus);
  int can_access = 0;
  TORCH_MUSA_CHECK(musaDeviceCanAccessPeer(&can_access, device, peer_device));
  return can_access != 0;
}

Allocator* getMUSADeviceAllocator() {
  return c10::musa::MUSACachingAllocator::get();
}

void registerMusaDeviceProperties(PyObject* module) {
  // Add _musaDeviceProperties class to torch_musa._MUSAC.
  auto py_module = pybind11::handle(module).cast<pybind11::module>();

  // Set musa version
  py_module.attr("_musa_version") = py::str(std::to_string(MUSA_VERSION));

  py::class_<musaDeviceProp>(py_module, "_MusaDeviceProperties")
      .def_readonly("name", &musaDeviceProp::name)
      .def_readonly("major", &musaDeviceProp::major)
      .def_readonly("minor", &musaDeviceProp::minor)
      .def_readonly("is_multi_gpu_board", &musaDeviceProp::isMultiGpuBoard)
      .def_readonly("is_integrated", &musaDeviceProp::integrated)
      .def_readonly(
          "multi_processor_count", &musaDeviceProp::multiProcessorCount)
      .def_readonly("total_memory", &musaDeviceProp::totalGlobalMem)
      .def("__repr__", [](const musaDeviceProp& prop) {
        std::ostringstream stream;
        stream << "_MusaDeviceProperties(name='" << prop.name << "', major='"
               << prop.major << ", minor=" << prop.minor
               << ", total_memory=" << prop.totalGlobalMem / (1024 * 1024)
               << "MB, multi_processor_count=" << prop.multiProcessorCount
               << ")";
        return stream.str();
      });
}

} // namespace musa
} // namespace at
