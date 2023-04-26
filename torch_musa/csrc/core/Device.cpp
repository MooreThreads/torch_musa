#include "torch_musa/csrc/core/Device.h"
#include <c10/util/CallOnce.h>
#include <deque>
#include "torch_musa/csrc/core/MUSAException.h"

namespace torch_musa {

namespace {

DeviceIndex num_mtgpus = -1;
c10::once_flag init_flag;
std::deque<c10::once_flag> device_flags;
std::vector<musaDeviceProp> device_properties;

void initMUSAContextVectors() {
  num_mtgpus = device_count();
  device_flags.resize(num_mtgpus);
  device_properties.resize(num_mtgpus);
}

void initDeviceProperty(DeviceIndex device_index) {
  musaDeviceProp device_prop;
  TORCH_MUSA_CHECK(musaGetDeviceProperties(&device_prop, device_index));
  device_properties[device_index] = device_prop;
}

} // anonymous namespace

DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      int result;
      TORCH_MUSA_CHECK(musaGetDeviceCount(&result));
      TORCH_INTERNAL_ASSERT(
          result <= std::numeric_limits<DeviceIndex>::max(),
          "Too many MUSA devices, DeviceIndex overflowed");
      return result;
    } catch (const c10::Error& ex) {
      // Terminated if fail and log the following message.
      TORCH_INTERNAL_ASSERT(false, "MUSA initialization: ", ex.msg());
    }
  }();

  return static_cast<DeviceIndex>(count);
}

DeviceIndex current_device() {
  int cur_device;
  TORCH_MUSA_CHECK(musaGetDevice(&cur_device));
  return static_cast<DeviceIndex>(cur_device);
}

void set_device(DeviceIndex device) {
  TORCH_MUSA_CHECK(musaSetDevice(static_cast<int>(device)));
}

DeviceIndex exchangeDevice(DeviceIndex device) {
  if (device < 0) {
    return static_cast<DeviceIndex>(-1);
  }
  auto cur_device = current_device();
  if (cur_device != device) {
    set_device(device);
  }
  return cur_device;
}

musaDeviceProp* getDeviceProperties(int device) {
  c10::call_once(init_flag, initMUSAContextVectors);
  if (device == -1)
    device = current_device();
  AT_ASSERT(device >= 0 && device < num_mtgpus);
  c10::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

bool canDeviceAccessPeer(int device, int peer_device) {
  c10::call_once(init_flag, initMUSAContextVectors);
  if (device == -1)
    device = current_device();
  AT_ASSERT(device >= 0 && device < num_mtgpus);
  AT_ASSERT(peer_device >= 0 && peer_device < num_mtgpus);
  int can_access = 0;
  TORCH_MUSA_CHECK(musaDeviceCanAccessPeer(&can_access, device, peer_device));
  return can_access != 0;
}

void Synchronize() {
  musaDeviceSynchronize();
}

void registerMusaDeviceProperties(PyObject* module) {
  // Add _musaDeviceProperties class to torch_musa._MUSAC.
  auto py_module = pybind11::handle(module).cast<pybind11::module>();
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

} // namespace torch_musa
