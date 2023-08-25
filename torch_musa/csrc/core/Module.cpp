#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/invalid_arguments.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <vector>

#include "torch_musa/csrc/amp/autocast_mode.h"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Allocator.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/Event.h"
#include "torch_musa/csrc/core/PythonTensor.h"
#include "torch_musa/csrc/core/Sleep.h"
#include "torch_musa/csrc/core/Stream.h"
#ifdef USE_MCCL
#include "torch_musa/csrc/distributed/Register.h"
#endif
#include "torch_musa/csrc/utils/musa_lazy_init.h"

#include "torch_musa/csrc/utils/Logging.h"
#include "torch_musa/csrc/utils/musa_lazy_init.h"

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

PyObject* PyMusaEmptyCache(PyObject* unused0, PyObject* unused1) {
  HANDLE_TH_ERRORS
  c10::musa::MUSACachingAllocator::EmptyCache();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaResetPeakStats(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS
  c10::musa::MUSACachingAllocator::ResetPeakStats();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaMemoryStats(PyObject* /* unused */, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  int device = (int)THPUtils_unpackLong(arg);

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

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaMemorySnapshot(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS
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

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

static void BindGetDeviceProperties(PyObject* module) {
  // Add method to torch_musa
  auto m = py::handle(module).cast<py::module>();
  m.def(
      "_get_device_properties",
      [](int64_t device) -> musaDeviceProp* {
        return at::musa::getDeviceProperties(device);
      },
      py::return_value_policy::reference);
}

static PyObject* PyMusaInitExtension(
    PyObject* /* unused */,
    PyObject* /* unused */) {
  HANDLE_TH_ERRORS
#if C10_ASAN_ENABLED
  TORCH_WARN(
      "torch.cuda: your pytorch binary has address sanitizer (asan) built in, "
      "asan is currently not compatible with torch.cuda module, "
      "you might get unexpected behavior (eg. out of memory, crash, etc.), "
      "please rebuild pytorch without asan if you need to use this module");
#endif
  at::musa::lazyInitMUSA();
  auto m = THPObjectPtr(PyImport_ImportModule("torch_musa"));
  if (!m)
    throw python_error();

  bool has_half = true;

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  set_module_attr("has_half", has_half ? Py_True : Py_False);

  const int64_t num_gpus = c10::musa::device_count();
  auto default_musa_generators = PyTuple_New(static_cast<Py_ssize_t>(num_gpus));
  for (const auto i : c10::irange(num_gpus)) {
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(
        at::musa::detail::getDefaultMUSAGenerator(i));
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_musa_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_musa_generators);
  BindGetDeviceProperties(m);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

#ifdef USE_MCCL
void AddMusaProcessGroupMethods(PyObject* module) {
  registerProcessGroupMCCL(module);
}
#endif

PyObject* PyMusaSetDevice(PyObject* /* unused */, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to setDevice");
  int64_t device = THPUtils_unpackLong(arg);
  torch::utils::musa_lazy_init();
  c10::musa::set_device(device);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaGetDevice(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS
  torch::utils::musa_lazy_init();
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  auto device = static_cast<int>(c10::musa::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaCanDeviceAccessPeer(PyObject* /* unused */, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* arg1 = nullptr;
  PyObject* arg2 = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "can_device_peer_access",
        1,
        "(int device, int peer_device);");
    return nullptr;
  }
  THPUtils_assert(
      THPUtils_checkLong(arg1), "invalid argument to canDeviceAccessPeer");
  THPUtils_assert(
      THPUtils_checkLong(arg2), "invalid argument to canDeviceAccessPeer");
  int64_t device = THPUtils_unpackLong(arg1);
  int64_t peer_device = THPUtils_unpackLong(arg2);
  torch::utils::musa_lazy_init();
  auto can_access = at::musa::canDeviceAccessPeer(device, peer_device);
  return PyBool_FromLong(can_access);
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaGetDeviceCount(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(at::musa::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaExchangeDevice(PyObject* /* unused */, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  int64_t device = THPUtils_unpackLong(arg);
  torch::utils::musa_lazy_init();
  auto exchanged_device = at::musa::exchangeDevice(device);
  return THPUtils_packInt32(exchanged_device);
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaSynchronize(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS
  c10::musa::Synchronize();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaGetDefaultStream(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(device_index), "invalid argument to getDefaultStream");
  int64_t device = THPUtils_unpackLong(device_index);
  auto stream = at::musa::getDefaultMUSAStream(device);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple,
      1,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaGetCurrentStream(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  int64_t device = THPUtils_unpackLong(device_index);
  auto stream = at::musa::getCurrentMUSAStream(device);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple,
      1,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaGetCurrentRawStream(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromVoidPtr(at::musa::getCurrentMUSAStream(device).stream());
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaSetStream(
    PyObject* /* unused */,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "stream_id", "device_index", "device_type", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|LLL",
          const_cast<char**>(kwlist),
          &stream_id,
          &device_index,
          &device_type)) {
  }

  auto stream = at::musa::MUSAStream::unpack3(
      stream_id, device_index, static_cast<c10::DeviceType>(device_type));

  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  auto device = static_cast<int>(c10::musa::current_device());
  if (device != stream.device_index()) {
    c10::musa::set_device(stream.device_index());
  }
  at::musa::setCurrentMUSAStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaSleep(PyObject* /* unused */, PyObject* cycles) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(cycles), "torch.musa._sleep(): expected 'int'");
  at::musa::sleep(THPUtils_unpackLong(cycles));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaMudnnVersion(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(MUDNN_VERSION);
  END_HANDLE_TH_ERRORS
}

static PyMethodDef MusaMemoryMethods[] = {
    {"_musa_emptyCache", PyMusaEmptyCache, METH_NOARGS, nullptr},
    {"_musa_memoryStats", PyMusaMemoryStats, METH_O, nullptr},
    {"_musa_resetPeakStats", PyMusaResetPeakStats, METH_NOARGS, nullptr},
    {"_musa_memorySnapshot", PyMusaMemorySnapshot, METH_NOARGS, nullptr},
    {nullptr}};

static PyMethodDef MusaStreamMethods[] = {
    {"_musa_getDefaultStream", PyMusaGetDefaultStream, METH_O, nullptr},
    {"_musa_getCurrentStream", PyMusaGetCurrentStream, METH_O, nullptr},
    {"_musa_getCurrentRawStream", PyMusaGetCurrentRawStream, METH_O, nullptr},
    {"_musa_setStream",
     castPyCFunctionWithKeywords(PyMusaSetStream),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_musa_sleep", PyMusaSleep, METH_O, nullptr},
    {nullptr}};

static PyMethodDef MusaDeviceMethods[] = {
    {"_musa_init", PyMusaInitExtension, METH_NOARGS, nullptr},
    {"_musa_getDevice", PyMusaGetDevice, METH_NOARGS, nullptr},
    {"_musa_getDeviceCount", PyMusaGetDeviceCount, METH_NOARGS, nullptr},
    {"_musa_setDevice", PyMusaSetDevice, METH_O, nullptr},
    {"_musa_exchangeDevice", PyMusaExchangeDevice, METH_O, nullptr},
    {"_musa_canDeviceAccessPeer",
     PyMusaCanDeviceAccessPeer,
     METH_VARARGS,
     nullptr},
    {"_musa_synchronize", PyMusaSynchronize, METH_NOARGS, nullptr},
    {"_mudnn_version", PyMusaMudnnVersion, METH_NOARGS, nullptr},
    {nullptr}};

PyObject* module;
static std::vector<PyMethodDef> methods;

PyObject* InitMusaModule() {
  at::internal::lazy_init_num_threads();

  // Initialize some Python bindings.
  torch::musa::InitializePythonBindings();

  AddPyMethodDefs(methods, torch::musa::GetTensorMethods());
  AddPyMethodDefs(methods, MusaDeviceMethods);
  AddPyMethodDefs(methods, MusaStreamMethods);
  AddPyMethodDefs(methods, MusaMemoryMethods);
  AddPyMethodDefs(methods, at::musa::autocast::GetAutocastMethods());

  static struct PyModuleDef musa_module = {
      PyModuleDef_HEAD_INIT, "torch_musa._MUSAC", nullptr, -1, methods.data()};
  module = PyModule_Create(&musa_module);

  THMPStream_init(module);
  THMPEvent_init(module);

#ifdef USE_MCCL
  AddMusaProcessGroupMethods(module);
#endif
  // Register MUSA device properties
  at::musa::registerMusaDeviceProperties(module);

  return module;
}
