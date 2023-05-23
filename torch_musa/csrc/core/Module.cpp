#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/invalid_arguments.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <vector>

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Allocator.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/Event.h"
#include "torch_musa/csrc/core/MUSAHooksInterface.h"
#include "torch_musa/csrc/core/Sleep.h"
#include "torch_musa/csrc/core/Stream.h"

// yang.zhao: copied from torch/csrc/utils.cpp to avoid including other things.
void THPUtils_invalidArguments(
    PyObject* given_args,
    PyObject* given_kwargs,
    const char* function_name,
    size_t num_options,
    ...) {
  std::vector<std::string> option_strings;
  va_list option_list;
  va_start(option_list, num_options);
  std::generate_n(
      std::back_inserter(option_strings), num_options, [&option_list] {
        return va_arg(option_list, const char*);
      });
  va_end(option_list);

  PyErr_SetString(
      PyExc_TypeError,
      torch::format_invalid_args(
          given_args, given_kwargs, function_name, option_strings)
          .c_str());
}

py::object PyMusa_EmptyCache() {
  c10::musa::MUSACachingAllocator::EmptyCache();
  return py::none();
}

py::object PyMusa_ResetPeakStats() {
  c10::musa::MUSACachingAllocator::ResetPeakStats();
  return py::none();
}

py::object PyMusa_MemoryStats(int64_t device) {
  using c10::musa::MUSACachingAllocator::DeviceStats;
  using c10::musa::MUSACachingAllocator::Stat;
  using c10::musa::MUSACachingAllocator::StatArray;
  using c10::musa::MUSACachingAllocator::StatType;

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
        statTypeNames = {"all", "small_pool", "large_pool"};
    py::dict dict;
    for (size_t i = 0; i < statTypeNames.size(); ++i) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const DeviceStats stats =
      c10::musa::MUSACachingAllocator::GetDeviceStats(device);

  py::dict result;
  result["num_alloc_retries"] = stats.num_alloc_retries;
  result["num_ooms"] = stats.num_ooms;
  result["max_split_size"] = stats.max_split_size;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
  result["oversize_allocations"] = statToDict(stats.oversize_allocations);
  result["oversize_segments"] = statToDict(stats.oversize_segments);

  return result;
}

py::object PyMusa_MemorySnapshot() {
  using c10::musa::MUSACachingAllocator::BlockInfo;
  using c10::musa::MUSACachingAllocator::SegmentInfo;

  const auto segmentInfoToDict = [](const SegmentInfo& segmentInfo) {
    py::dict segmentDict;
    segmentDict["device"] = segmentInfo.device;
    segmentDict["address"] = segmentInfo.address;
    segmentDict["total_size"] = segmentInfo.total_size;
    segmentDict["allocated_size"] = segmentInfo.allocated_size;
    segmentDict["active_size"] = segmentInfo.active_size;
    segmentDict["segment_type"] = (segmentInfo.is_large ? "large" : "small");

    py::list blocks;
    for (const auto& blockInfo : segmentInfo.blocks) {
      py::dict blockDict;
      blockDict["size"] = blockInfo.size;
      blockDict["state"] =
          (blockInfo.allocated
               ? "active_allocated"
               : (blockInfo.active ? "active_pending_free" : "inactive"));
      blocks.append(blockDict);
    }
    segmentDict["blocks"] = blocks;

    return segmentDict;
  };

  const auto& snapshot = c10::musa::MUSACachingAllocator::GetMemorySnapshot();
  py::list result;

  for (const auto& segmentInfo : snapshot) {
    result.append(segmentInfoToDict(segmentInfo));
  }

  return result;
}

static py::object THMPModule_initExtension() {
  static c10::once_flag thm_init;
  c10::call_once(thm_init, [] { at::detail::getMUSAHooks().initMUSA(); });
  return py::none();
}

void AddMusaDeviceMethods(PyObject* module) {
  auto py_module = py::reinterpret_borrow<py::module>(module);

  // init musa
  py_module.def("_musa_init", []() { THMPModule_initExtension(); });

  py_module.def(
      "_musa_getDeviceCount", []() { return c10::musa::device_count(); });
  py_module.def(
      "_musa_getDevice", []() { return c10::musa::current_device(); });
  py_module.def(
      "_musa_setDevice", [](int64_t device) { c10::musa::set_device(device); });
  py_module.def("_musa_exchangeDevice", [](int64_t device) {
    return c10::musa::exchangeDevice(device);
  });
  py_module.def(
      "_musa_canDeviceAccessPeer", [](int64_t device, int64_t peer_device) {
        return c10::musa::canDeviceAccessPeer(device, peer_device);
      });
  py_module.def(
      "_get_device_properties",
      [](int64_t device) -> musaDeviceProp* {
        return c10::musa::getDeviceProperties(device);
      },
      py::return_value_policy::reference);

  // Synchronize musa device.
  py_module.def("_musa_synchronize", []() { c10::musa::Synchronize(); });
}

void AddMusaStreamMethods(PyObject* module) {
  auto py_module = py::reinterpret_borrow<py::module>(module);

  py_module.def("_musa_getDefaultStream", [](int64_t device_index) {
    auto stream = c10::musa::getDefaultMUSAStream(device_index);
    return py::make_tuple(
        static_cast<int64_t>(stream.id()),
        static_cast<int64_t>(stream.device_index()),
        static_cast<int64_t>(stream.device_type()));
  });
  py_module.def("_musa_getCurrentStream", [](int64_t device_index) {
    auto stream = c10::musa::getCurrentMUSAStream(device_index);
    return py::make_tuple(
        static_cast<int64_t>(stream.id()),
        static_cast<int64_t>(stream.device_index()),
        static_cast<int64_t>(stream.device_type()));
  });
  py_module.def("_musa_getCurrentRawStream", [](int64_t device_index) {
    return c10::musa::getCurrentMUSAStream(device_index);
  });
  py_module.def("_musa_setStream", [](py::kwargs& stream_attr) {
    int64_t device_type = stream_attr["device_type"].cast<int64_t>();
    int64_t stream_id = stream_attr["stream_id"].cast<int64_t>();
    int64_t device_index = stream_attr["device_index"].cast<int64_t>();
    auto stream = c10::musa::MUSAStream::unpack3(
        stream_id, device_index, static_cast<c10::DeviceType>(device_type));
    auto device = static_cast<int>(c10::musa::current_device());
    if (device != stream.device_index()) {
      c10::musa::set_device(stream.device_index());
    }
    c10::musa::setCurrentMUSAStream(stream);
  });

  // Sleep function
  py_module.def("_musa_sleep", [](int64_t cycles) { at::musa::sleep(cycles); });
}

void AddMusaMemoryMethods(PyObject* module) {
  auto py_module = py::reinterpret_borrow<py::module>(module);

  py_module.def("_musa_emptyCache", []() { return PyMusa_EmptyCache(); });
  py_module.def(
      "_musa_resetPeakStats", []() { return PyMusa_ResetPeakStats(); });
  py_module.def("_musa_memoryStats", [](int64_t device) {
    return PyMusa_MemoryStats(device);
  });
  py_module.def(
      "_musa_memorySnapshot", []() { return PyMusa_MemorySnapshot(); });
}

void InitMusaModule(PyObject* module) {
  // TODO(mt-ai) Let's lazily init musa devices first.
  THMPStream_init(module);
  THMPEvent_init(module);
  auto py_module = py::reinterpret_borrow<py::module>(module);

  AddMusaDeviceMethods(module);
  AddMusaStreamMethods(module);
  AddMusaMemoryMethods(module);

  // Register MUSA device properties
  c10::musa::registerMusaDeviceProperties(module);
}
