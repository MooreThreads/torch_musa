#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <vector>

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Allocator.h"
#include "torch_musa/csrc/core/Device.h"

void AddPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods) {
  if (!vector.empty()) {
    // remove nullptr terminator
    vector.pop_back();
  }
  while (true) {
    vector.push_back(*methods);
    if (!methods->ml_name) {
      break;
    }
    methods++;
  }
}

PyObject* PyMusa_EmptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  musa::MUSACachingAllocator::EmptyCache();
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* PyMusa_MemoryStats(PyObject* unused0, PyObject* unused1) {
  HANDLE_TH_ERRORS

  using musa::MUSACachingAllocator::DeviceStats;
  using musa::MUSACachingAllocator::Stat;
  using musa::MUSACachingAllocator::StatArray;
  using musa::MUSACachingAllocator::StatType;

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

  const DeviceStats stats = musa::MUSACachingAllocator::GetDeviceStats();

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

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusa_MemorySnapshot(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS

  using musa::MUSACachingAllocator::BlockInfo;
  using musa::MUSACachingAllocator::SegmentInfo;

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

  const auto& snapshot = musa::MUSACachingAllocator::GetMemorySnapshot();
  py::list result;

  for (const auto& segmentInfo : snapshot) {
    result.append(segmentInfoToDict(segmentInfo));
  }

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

static PyMethodDef TorchMusaMethods[] = {
    {"_musa_emptyCache", PyMusa_EmptyCache, METH_NOARGS, nullptr},
    {"_musa_memoryStats", PyMusa_MemoryStats, METH_NOARGS, nullptr},
    {"_musa_memorySnapshot", PyMusa_MemorySnapshot, METH_NOARGS, nullptr}};

PyObject* module;

static std::vector<PyMethodDef> methods;

PyObject* InitMusaModule() {
  HANDLE_TH_ERRORS
  at::internal::lazy_init_num_threads();

  AddPyMethodDefs(methods, TorchMusaMethods);

  static struct PyModuleDef musa_module = {
      PyModuleDef_HEAD_INIT, "torch_musa._MUSAC", nullptr, -1, methods.data()};
  module = PyModule_Create(&musa_module);

  // TODO(mt-ai) we need to have an init function for musa device.
  THPDevice_init(module);
  auto py_module = py::reinterpret_borrow<py::module>(module);

  // Device Management
  py_module.def(
      "_musa_getDeviceCount", []() { return torch_musa::device_count(); });
  py_module.def(
      "_musa_getDevice", []() { return torch_musa::current_device(); });
  py_module.def(
      "_musa_setDevice", [](int device) { torch_musa::set_device(device); });
  // Synchronize musa device.
  py_module.def("_musa_synchronize", []() { at::native::musa::Synchronize(); });

  return module;
  END_HANDLE_TH_ERRORS
}

PyMODINIT_FUNC PyInit__MUSAC(void) {
  return InitMusaModule();
}

#pragma GCC diagnostic pop
