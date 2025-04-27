#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <ATen/autocast_mode.h>
#include <c10/util/Backtrace.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/profiler/python/combined_traceback.h>
#include <torch/csrc/utils/invalid_arguments.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <cstddef>
#include <vector>

#include "torch_musa/csrc/amp/autocast_mode.h"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/Event.h"
#include "torch_musa/csrc/core/MUSAAllocatorConfig.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"
#include "torch_musa/csrc/core/PythonTensor.h"
#include "torch_musa/csrc/core/Sleep.h"
#include "torch_musa/csrc/core/Stream.h"
#ifdef USE_MCCL
#include "torch_musa/csrc/core/PythonComm.h"
#include "torch_musa/csrc/distributed/Register.h"
#endif
#include "torch_musa/csrc/aten/musa/MUSAGraphsUtils.muh"
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/core/MusaIPCTypes.h"
#include "torch_musa/csrc/core/Storage.h"
#include "torch_musa/csrc/core/StorageSharing.h"
#include "torch_musa/csrc/utils/Logging.h"
#include "torch_musa/csrc/utils/musa_lazy_init.h"

#include "torch_musa/csrc/aten/ConvUtils.h"
#include "torch_musa/csrc/core/MemorySnapshot.h"

#include <pthread.h>

bool in_bad_fork = false; // True for children forked after musa init

// Called in the forked child if musa has already been initialized
static void forked_child() {
  in_bad_fork = true;
  torch::utils::set_requires_musa_init(true);
}

static void poison_fork() {
  static c10::once_flag flag;
  c10::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
}

/**
 * @brief Forward declaration of `THCPGraph_init`, which is defined at Graph.cpp
 */
void THCPGraph_init(PyObject* module);

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
  c10::musa::MUSACachingAllocator::emptyCache();

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaResetPeakMemoryStats(PyObject* /* unused */, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
  const int device = (int)THPUtils_unpackLong(arg);
  c10::musa::MUSACachingAllocator::resetPeakStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* PyMusaMemoryStats(PyObject* /* unused */, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  const int device = (int)THPUtils_unpackLong(arg);

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
    for (const auto i : c10::irange(statTypeNames.size())) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const DeviceStats stats =
      c10::musa::MUSACachingAllocator::getDeviceStats(device);

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
  result["requested_bytes"] = statArrayToDict(stats.requested_bytes);
  result["oversize_allocations"] = statToDict(stats.oversize_allocations);
  result["oversize_segments"] = statToDict(stats.oversize_segments);

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

struct Frame {
  PyCodeObject* code;
  int lasti;
};

torch::CapturedTraceback* getFromContext(
    const std::shared_ptr<c10::GatheredContext>& x) {
  if (torch::CapturedTraceback* sc =
          dynamic_cast<torch::CapturedTraceback*>(x.get())) {
    return sc;
  }
  TORCH_CHECK(
      false,
      "attempting to gather stack context from the wrong StackContext type.");
}

PyObject* PyMusaMemorySnapshot(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS

  using c10::musa::MUSACachingAllocator::BlockInfo;
  using c10::musa::MUSACachingAllocator::SegmentInfo;

  py::str device_s = "device";
  py::str address_s = "address";
  py::str total_size_s = "total_size";
  py::str allocated_size_s = "allocated_size";
  py::str active_size_s = "active_size";
  py::str requested_size_s = "requested_size";
  py::str stream_s = "stream";
  py::str segment_type_s = "segment_type";
  py::str segment_pool_id = "segment_pool_id";
  py::str large_s = "large";
  py::str small_s = "small";
  py::str size_s = "size";
  py::str state_s = "state";
  py::str active_allocated_s = "active_allocated";
  py::str active_pending_free_s = "active_pending_free";
  py::str inactive_s = "inactive";
  py::str addr_s = "addr";
  py::str cpp_frames_s = "cpp_frames";
  py::str blocks_s = "blocks";
  py::str is_expandable_s = "is_expandable";
  py::str frames_s = "frames";
  py::str time_us_s = "time_us";

  py::list empty_frames;
  std::vector<torch::CapturedTraceback*> to_gather_frames;
  std::vector<py::dict> to_gather_dest;

  auto add_frame_key = [&](const py::dict& d,
                           const std::shared_ptr<c10::GatheredContext>& ctx) {
    if (ctx) {
      auto sc = getFromContext(ctx);
      to_gather_frames.emplace_back(sc);
      to_gather_dest.emplace_back(d);
    } else {
      d[frames_s] = empty_frames;
    }
  };

  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    py::dict segmentDict;
    segmentDict[device_s] = segmentInfo.device;
    segmentDict[address_s] = segmentInfo.address;
    segmentDict[total_size_s] = segmentInfo.total_size;
    segmentDict[allocated_size_s] = segmentInfo.allocated_size;
    segmentDict[active_size_s] = segmentInfo.active_size;
    segmentDict[requested_size_s] = segmentInfo.requested_size;
    // we want the python objects to pickle easily so use an int to
    // represent the stream rather than a torch.musa.stream object
    segmentDict[stream_s] = int64_t(segmentInfo.stream);
    segmentDict[segment_type_s] = (segmentInfo.is_large ? large_s : small_s);
    segmentDict[segment_pool_id] = segmentInfo.owner_private_pool_id;
    segmentDict[is_expandable_s] = segmentInfo.is_expandable;
    add_frame_key(segmentDict, segmentInfo.context_when_allocated);

    auto address = segmentInfo.address;
    py::list blocks;
    for (const auto& blockInfo : segmentInfo.blocks) {
      py::dict blockDict;
      blockDict[address_s] = address;
      blockDict[size_s] = blockInfo.size;
      blockDict[requested_size_s] = blockInfo.requested_size;
      blockDict[state_s] =
          (blockInfo.allocated
               ? active_allocated_s
               : (blockInfo.active ? active_pending_free_s : inactive_s));
      add_frame_key(blockDict, blockInfo.context_when_allocated);
      blocks.append(blockDict);
      address += blockInfo.size;
    }
    segmentDict[blocks_s] = blocks;

    return segmentDict;
  };

  const auto& snapshot = c10::musa::MUSACachingAllocator::snapshot();
  py::list segments;

  for (const auto& segmentInfo : snapshot.segments) {
    segments.append(segmentInfoToDict(segmentInfo));
  }

  py::list traces;
  py::str action_s = "action";
  py::str alloc_s = "alloc";
  py::str free_requested_s = "free_requested";
  py::str free_completed_s = "free_completed";
  py::str segment_alloc_s = "segment_alloc";
  py::str segment_free_s = "segment_free";
  py::str segment_map_s = "segment_map";
  py::str segment_unmap_s = "segment_unmap";

  py::str snapshot_s = "snapshot";
  py::str oom_s = "oom";
  py::str device_free_s = "device_free";

  using namespace c10::musa::MUSACachingAllocator;
  auto action_to_str = [&](TraceEntry::Action action) {
    switch (action) {
      case TraceEntry::ALLOC:
        return alloc_s;
      case TraceEntry::FREE_REQUESTED:
        return free_requested_s;
      case TraceEntry::FREE_COMPLETED:
        return free_completed_s;
      case TraceEntry::SEGMENT_ALLOC:
        return segment_alloc_s;
      case TraceEntry::SEGMENT_FREE:
        return segment_free_s;
      case TraceEntry::OOM:
        return oom_s;
      case TraceEntry::SNAPSHOT:
        return snapshot_s;
      case TraceEntry::SEGMENT_UNMAP:
        return segment_unmap_s;
      case TraceEntry::SEGMENT_MAP:
        return segment_map_s;
    }
    throw std::runtime_error("unreachable");
  };

  for (const auto& traceInfo : snapshot.device_traces) {
    py::list trace;
    for (const auto& te : traceInfo) {
      py::dict trace_entry;
      if (te.context_) {
        // without further compression frames can get really large on dump
        auto sc = getFromContext(te.context_);
        to_gather_frames.emplace_back(sc);
        to_gather_dest.emplace_back(trace_entry);
      }
      trace_entry[action_s] = action_to_str(te.action_);
      trace_entry[TraceEntry::OOM == te.action_ ? device_free_s : addr_s] =
          te.addr_;
      trace_entry[size_s] = te.size_;
      trace_entry[stream_s] = int64_t(te.stream_);
      trace_entry[time_us_s] = te.time_.t_;
      trace.append(trace_entry);
    }
    traces.append(trace);
  }

  py::dict result;
  result["segments"] = segments;
  result["device_traces"] = traces;

  // py_symbolize is local, so we patched combined_traceback.h to make it
  // visible for torch_musa
  auto frames = torch::py_symbolize(to_gather_frames);
  for (auto i : c10::irange(frames.size())) {
    to_gather_dest.at(i)[frames_s] = frames.at(i);
  }

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaAttachOutOfMemoryObserver(
    PyObject* _unused,
    PyObject* observer) {
  HANDLE_TH_ERRORS
  Py_XINCREF(observer);
  auto obs = [observer](
                 int64_t device,
                 int64_t alloc,
                 int64_t device_allocated,
                 int64_t device_free) {
    py::gil_scoped_acquire g;
    PyObject* result = PyObject_CallFunction(
        observer, "LLLL", device, alloc, device_allocated, device_free);
    if (!result) {
      throw py::error_already_set();
    }
    Py_XDECREF(result);
  };
  at::musa::lazyInitMUSA();
  c10::musa::MUSACachingAllocator::attachOutOfMemoryObserver(std::move(obs));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaSetMemoryFraction(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* fraction_o = nullptr;
  PyObject* device_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &fraction_o, &device_o)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "set_memory_fraction",
        1,
        "(double fraction, int device);");
    return nullptr;
  }
  double fraction = PyFloat_AsDouble(fraction_o);
  int64_t device = PyLong_AsLongLong(device_o);

  c10::musa::MUSACachingAllocator::setMemoryFraction(fraction, device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* PyMusaresetPeakMemoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
  const int device = (int)THPUtils_unpackLong(arg);
  c10::musa::MUSACachingAllocator::resetPeakStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* PyMusasetMusaCachingAllocatorSettings(
    PyObject* _unused,
    PyObject* env) {
  HANDLE_TH_ERRORS
  c10::musa::MUSACachingAllocator::setAllocatorSettings(
      THPUtils_unpackString(env));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static void RegisterMusaDeviceProperties(PyObject* module) {
  // Add _musaDeviceProperties class to torch_musa._MUSAC.
  auto m = py::handle(module).cast<py::module>();

  // Set musa version
#if defined(REAL_MUSA_VERSION)
  m.attr("_musa_version") = py::str(std::to_string(REAL_MUSA_VERSION));
#else
  m.attr("_musa_version") = py::str("0"); // invalid path ?
#endif
  py::class_<musaDeviceProp>(m, "_MusaDeviceProperties")
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
  m.def(
      "_musa_record_memory_history_legacy",
      static_cast<void (*)(bool, bool, int64_t, bool, bool)>(
          torch::musa::_record_memory_history));
  m.def(
      "_musa_record_memory_history",
      static_cast<void (*)(
          std::optional<std::string>,
          std::optional<std::string>,
          std::string,
          size_t)>(torch::musa::_record_memory_history));
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
#if C10_ASAN_ENABLED
  TORCH_WARN(
      "torch.musa: your pytorch binary has address sanitizer (asan) built in, "
      "asan is currently not compatible with torch.musa module, "
      "you might get unexpected behavior (eg. out of memory, crash, etc.), "
      "please rebuild pytorch without asan if you need to use this module");
#endif
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(!in_bad_fork); // Handled at python level
  poison_fork();
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
  poison_fork();
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

PyObject* PyMusaIPCCollect(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  bool freed_memory = torch::musa::MusaIPCCollect();
  return PyBool_FromLong(freed_memory);
  END_HANDLE_TH_ERRORS
}

static PyObject* PyMusaIsInBadFork(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaGetArchFlags(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  poison_fork();
#ifdef MUSA_ARCH_FLAGS
  static const char* flags = C10_STRINGIZE(MUSA_ARCH_FLAGS);
  return THPUtils_packString(flags);
#else
  Py_RETURN_NONE;
#endif
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

PyObject* THCPModule_isCurrentStreamCapturing_wrap(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // If there's no musa context, at::musa::currentStreamCaptureStatus returns
  // CaptureStatus::None without initializing a context.
  if (at::musa::currentStreamCaptureStatus() == at::musa::CaptureStatus::None) {
    Py_RETURN_FALSE;
  } else {
    Py_RETURN_TRUE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaMudnnVersion(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(MUDNN_VERSION);
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaSetDefaultDtype(PyObject* _unused, PyObject* type) {
  HANDLE_TH_ERRORS
  torch::musa::PySetDefaultDtype(type);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

at::MemoryFormat DetermineBackendMemoryFormat(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::native::ConvBackend backend) {
  at::MemoryFormat backend_memory_format = at::MemoryFormat::Contiguous;
  auto k = weight.ndimension();

  switch (backend) {
    // Cudnn and Miopen backend are removed
    case at::native::ConvBackend::Mkldnn:
    case at::native::ConvBackend::MkldnnTranspose:
      if (at::native::mkldnn_conv_use_channels_last(input, weight)) {
        backend_memory_format = (k == 5) ? at::MemoryFormat::ChannelsLast3d
                                         : at::MemoryFormat::ChannelsLast;
      }
      break;
    case at::native::ConvBackend::Slow2d:
    case at::native::ConvBackend::SlowDilated2d:
    case at::native::ConvBackend::SlowTranspose2d:
      if (at::native::thnn_conv_use_channels_last(input, weight)) {
        backend_memory_format = at::MemoryFormat::ChannelsLast;
      }
      break;
    case at::native::ConvBackend::Overrideable:
      if (at::musa::musa_conv_use_channels_last(input, weight)) {
        backend_memory_format = (k == 5) ? at::MemoryFormat::ChannelsLast3d
                                         : at::MemoryFormat::ChannelsLast;
      }
      break;
    default:
      backend_memory_format = at::MemoryFormat::Contiguous;
  }

  return backend_memory_format;
}

static PyMethodDef MusaMemoryMethods[] = {
    {"_musa_emptyCache", PyMusaEmptyCache, METH_NOARGS, nullptr},
    {"_musa_memoryStats", PyMusaMemoryStats, METH_O, nullptr},
    {"_musa_memorySnapshot", PyMusaMemorySnapshot, METH_NOARGS, nullptr},
    {"_musa_attach_out_of_memory_observer",
     PyMusaAttachOutOfMemoryObserver,
     METH_O,
     nullptr},
    {"_musa_setMemoryFraction", PyMusaSetMemoryFraction, METH_VARARGS, nullptr},
    {"_musa_resetPeakMemoryStats", PyMusaresetPeakMemoryStats, METH_O, nullptr},
    {"_musa_setMusaCachingAllocatorSettings",
     PyMusasetMusaCachingAllocatorSettings,
     METH_O,
     nullptr},
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
    {"_musa_ipc_collect", PyMusaIPCCollect, METH_NOARGS, nullptr},
    {"_musa_isInBadFork", PyMusaIsInBadFork, METH_NOARGS, nullptr},
    {"_musa_isCurrentStreamCapturing",
     THCPModule_isCurrentStreamCapturing_wrap,
     METH_NOARGS,
     nullptr},
    {"_musa_getArchFlags", PyMusaGetArchFlags, METH_NOARGS, nullptr},
    {nullptr}};

static PyMethodDef MusaDtypeMethods[] = {
    {"_set_default_dtype", PyMusaSetDefaultDtype, METH_O, nullptr},
    {nullptr}};

PyObject* module;
static std::vector<PyMethodDef> methods;

PyObject* InitMusaModule() {
  at::internal::lazy_init_num_threads();

  // Initialize some Python bindings.
  torch::musa::InitializePythonBindings();

  AddPyMethodDefs(methods, torch::musa::GetTensorMethods());
  AddPyMethodDefs(methods, GetStorageSharingMethods());
  AddPyMethodDefs(methods, MusaDeviceMethods);
  AddPyMethodDefs(methods, MusaStreamMethods);
  AddPyMethodDefs(methods, MusaMemoryMethods);
  AddPyMethodDefs(methods, MusaDtypeMethods);
  AddPyMethodDefs(methods, at::autocast::musa::GetAutocastMethods());
  AddPyMethodDefs(methods, at::musa::GetContextMethods());
  AddPyMethodDefs(methods, at::musa::GetStorageMethods());

  static struct PyModuleDef musa_module = {
      PyModuleDef_HEAD_INIT, "torch_musa._MUSAC", nullptr, -1, methods.data()};
  module = PyModule_Create(&musa_module);

  THMPStream_init(module);
  THMPEvent_init(module);
  THCPGraph_init(module);

#ifdef USE_MCCL
  torch::musa::python::InitCommMethods(module);
  AddMusaProcessGroupMethods(module);
#endif
  // Register MUSA device properties
  RegisterMusaDeviceProperties(module);

  // for methods that may rely on backend's behavior,
  // we will replace PyTorch's original implementation
  auto py_module = py::reinterpret_borrow<py::module>(module);
  py_module.def(
      "_conv_determine_backend_memory_format", DetermineBackendMemoryFormat);

  return module;
}
