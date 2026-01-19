#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <iostream>
#include <thread>
#include <tuple>

#include <c10/util/thread_name.h>
#include "mccl.h"
#include "torch_musa/csrc/aten/musa/MUSAGraph.h"
#include "torch_musa/csrc/core/MUSAGraphsC10Utils.h"
#include "torch_musa/csrc/distributed/ProcessGroupMCCL.h"

namespace c10d {

constexpr const char* const kMCCLAbortedCommStoreKey = "MCCLABORTEDCOMM";
const int64_t ProcessGroupMCCL::kWatchdogThreadSleepMillis = 10000;
const int64_t ProcessGroupMCCL::kWorkCleanupThreadSleepMillis = 1000;
const int64_t ProcessGroupMCCL::kHeartBeatThreadSleepMillis = 1000;
constexpr int64_t kWaitForAbortCommStoreKey = 1000;
constexpr int64_t kSynchronizeBusyWaitMillis = 10;
thread_local uint64_t ProcessGroupMCCL::mcclActiveGroupCounter_ = 0;

namespace { // DDP Helper functions

const std::map<at::ScalarType, mcclDataType_t> mcclDataType = {
    {at::kFloat, mcclFloat},
    {at::kInt, mcclInt32},
    {at::kChar, mcclInt8},
    {at::kByte, mcclUint8},
    {at::kLong, mcclInt64},
    {at::kHalf, mcclFloat16},
    {at::kDouble, mcclFloat64},
    {at::kBool, mcclUint8},
#if MCCL_BF16_SUPPORTED
    {at::kBFloat16, mcclBfloat16},
#if MCCL_FP8_SUPPORTED
    {at::kFloat8_e5m2, mcclFp8E5M2},
    {at::kFloat8_e4m3fn, mcclFp8E4M3},
#endif
#endif
};

const std::map<ReduceOp::RedOpType, mcclRedOp_t> mcclOp = {
    {ReduceOp::MAX, mcclMax},
    {ReduceOp::SUM, mcclSum},
    {ReduceOp::MIN, mcclMin},
    {ReduceOp::PRODUCT, mcclProd},
    {ReduceOp::AVG, mcclAvg},
};

mcclDataType_t getMcclDataType(at::ScalarType type) {
  auto it = mcclDataType.find(type);
  TORCH_CHECK(
      it != mcclDataType.end(),
      "Input tensor data type in not supported for MCCL process group: ",
      type);
  return it->second;
}

bool complexViewAsRealAllowed(const ReduceOp& reduceOp) {
  switch (reduceOp) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case ReduceOp::SUM:
      return true;
    case ReduceOp::AVG:
      return true;
    case ReduceOp::PREMUL_SUM:
      return true;
    case ReduceOp::UNUSED:
      return true;
    default:
      return false;
  }
  return false;
}

// TODO(yueran-tang): Not finished since we only support a few Ops.
mcclRedOp_t getMcclReduceOp(
    const ReduceOp& reduce_op,
    at::Tensor& input,
    const mcclDataType_t& data_type,
    const mcclComm_t& comm) {
  try {
    if (input.scalar_type() == at::kBool) {
      // SUM of kBool is the same as "OR" or "MAX" of Boolean.
      // TODO(yueran-tang): Bool Max is mccl style reduceOp, and we need to
      // check it on mccl.
      if (reduce_op == ReduceOp::SUM) {
        return mcclMax;
      }
      if (reduce_op == ReduceOp::AVG) {
        TORCH_CHECK(false, "Cannot use ReduceOp.AVG with Boolean inputs");
      }
    }
    return mcclOp.at(reduce_op);
  } catch (const std::out_of_range& e) {
    TORCH_CHECK(false, "Unexpected ReduceOp: ", reduce_op);
  }
}

// Get a key string from device
inline std::string getKeyFromDevice(at::Device& device) {
  return std::to_string(device.index());
}

inline at::DeviceIndex getIndexFromDeviceKey(const std::string& deviceKey) {
  // initialize the device index to -1, which is an invalid value.
  int index = -1;
  try {
    index = std::stoi(deviceKey);
  } catch (const std::invalid_argument& e) {
    LOG(ERROR) << c10::str(
        "Invalid deviceKey: ", deviceKey, ",", e.what(), ".");
  } catch (const std::out_of_range& e) {
    LOG(ERROR) << "Out of range: " << e.what();
  }
  return static_cast<at::DeviceIndex>(index);
}

std::string getKeySendRecv(int myRank, int peer) {
  int lowRank = myRank < peer ? myRank : peer;
  int highRank = myRank < peer ? peer : myRank;
  std::string sendRecvPair =
      std::to_string(lowRank) + ":" + std::to_string(highRank);
  return sendRecvPair;
}

// Get device from tensor
inline at::Device getDevice(at::Tensor& tensor) {
  return tensor.device();
}

void syncStream(
    at::Device& device,
    at::musa::MUSAEvent& mcclEvent,
    at::musa::MUSAStream& mcclStream) {
  mcclEvent.record(at::musa::getCurrentMUSAStream(device.index()));
  mcclEvent.block(mcclStream);
}

std::string buildMcclUniqueIdStr(const mcclUniqueId& mcclID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&mcclID);
  std::ostringstream oss;
  for (const auto i : c10::irange(MCCL_UNIQUE_ID_BYTES)) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

std::string getMcclAbortedCommStoreKey(const std::string& mcclIdStr) {
  return std::string(kMCCLAbortedCommStoreKey) + ":" + mcclIdStr;
}

std::string getExceptionMsgFromExceptionPtr(
    const std::exception_ptr& exceptionPtr) {
  TORCH_CHECK(exceptionPtr != nullptr);
  try {
    std::rethrow_exception(exceptionPtr);
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "Unknown exception type";
  }
}

inline void errorIfCapturingNonCapturableMCCL() {
  // Always success. MCCL has no old version capturable problems.
  // It's just a placeholder for MCCL -> MCCL Porting.
}

} // namespace

// Map from each communicator to its device index.
// This map is used when register/deregister cache segments from cache
// allocator. See design notes below:
// - Each segment should be registered only to the communicator on the
//   same device.
// - We cannot reuse devMCCLCommMap_ in each ProcessGroup because the key may be
//   ranks rather than device in point-to-point case.
// - This map has also to be maintained as global variable since the register
//   hooks are called outside the scope of any PG, thus we need traverse
//   communicators in all PGs.
static std::unordered_map<std::shared_ptr<MCCLComm>, int> mcclCommDevIdxMap;
static std::mutex mcclCommDevIdxMapMutex;
static bool allocatorHooksAttached = false;

std::atomic<bool> ProcessGroupMCCL::shouldDump_(false);

static void cacheAllocatorRegisterHook(
    const c10::musa::MUSACachingAllocator::TraceEntry& te) {
  // // Register after SEGMENT_ALLOC
  // if (te.action_ !=
  //     c10::musa::MUSACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC) {
  //   return;
  // }

  // std::lock_guard<std::mutex> lock(mcclCommDevIdxMapMutex);
  // for (auto& it : mcclCommDevIdxMap) {
  //   auto& mcclComm = it.first;
  //   auto& devIdx = it.second;
  //   if (te.device_ == devIdx) {
  //     // NOLINTNEXTLINE(performance-no-int-to-ptr)
  //     mcclComm->registerSegment(reinterpret_cast<void*>(te.addr_), te.size_);
  //   }
  // }
}

static void cacheAllocatorDeregisterHook(
    const c10::musa::MUSACachingAllocator::TraceEntry& te) {
  // // deregister before SEGMENT_FREE
  // if (te.action_ !=
  //     c10::musa::MUSACachingAllocator::TraceEntry::Action::SEGMENT_FREE) {
  //   return;
  // }

  // std::lock_guard<std::mutex> lock(mcclCommDevIdxMapMutex);
  // for (auto& it : mcclCommDevIdxMap) {
  //   auto& mcclComm = it.first;
  //   auto& devIdx = it.second;
  //   if (te.device_ == devIdx) {
  //     // NOLINTNEXTLINE(performance-no-int-to-ptr)
  //     mcclComm->deregisterSegment(reinterpret_cast<void*>(te.addr_));
  //   }
  // }
}

std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
getMCCLCommDumpMap() {
  std::unordered_map<
      std::string /* mcclUniqueID */,
      std::unordered_map<std::string, std::string> /* dump from this comm */>
      mcclDumpMap;
  // dump_mccl_trace is only called from the default PG (local_id_=0), but we
  // want to dump from all comms so we need to iterate over mcclCommDevIdxMap,
  // which is static
  std::vector<std::shared_ptr<MCCLComm>> allMCCLComms;
  // within the critical section, we don't want to dump while holding the lock
  // as dump might hang
  mcclCommDevIdxMapMutex.lock();
  for (auto& [mcclComm, _] : mcclCommDevIdxMap) {
    allMCCLComms.push_back(mcclComm);
  }
  mcclCommDevIdxMapMutex.unlock();
  for (auto& mcclComm : allMCCLComms) {
    std::string mcclUniqueIDStr = buildMcclUniqueIdStr(mcclComm->getMcclId());
    mcclDumpMap[mcclUniqueIDStr] = mcclComm->mcclCommDump();
  }
  return mcclDumpMap;
}

std::optional<std::function<void(std::function<void(const std::string&)>)>>&
get_cpp_trace_dumper() {
  static std::optional<
      std::function<void(std::function<void(const std::string&)>)>>
      dumper(std::nullopt);
  return dumper;
}

gil_checker_t& get_gil_checker() {
  static gil_checker_t gil_checker = nullptr;
  return gil_checker;
}

static std::future<bool> launchAsyncGilCheck() {
  std::promise<bool> resultPromise;
  std::future<bool> resultFuture = resultPromise.get_future();
  TORCH_CHECK(get_gil_checker(), "Can't check GIL with null GIL checker");
  std::thread workerThread([promise = std::move(resultPromise)]() mutable {
    c10::setThreadName("pt_mccl_gil_chk");

    try {
      auto& gil_checker = get_gil_checker();
      promise.set_value((*gil_checker)());
    } catch (...) {
      promise.set_exception(std::current_exception());
    }
  });

  // Detach the thread to allow it to run independently
  workerThread.detach();

  return resultFuture;
}

std::ostream& operator<<(
    std::ostream& output,
    const ProcessGroupMCCL::WorkMCCL& workMCCL) {
  std::string workInfo;
  workInfo = c10::str(
      "WorkMCCL(",
      "SeqNum=",
      workMCCL.seq_,
      ", OpType=",
      opTypeToString(workMCCL.opType_),
      ", NumelIn=",
      workMCCL.numelIn_,
      ", NumelOut=",
      workMCCL.numelOut_,
      ", Timeout(ms)=",
      workMCCL.opTimeout_.count(),
      ")");
  return output << workInfo;
}

ProcessGroupMCCL::WorkMCCL::WorkMCCL(
    std::string pgUID,
    std::string pgDesc,
    at::Device& device,
    int rank,
    OpType opType,
    uint64_t seq,
    bool isP2P,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputs,
    bool desyncDebug,
    bool enableTiming,
    bool musaEventCacheEnabled,
    DebugLevel distDebugLevel)
    : Work(rank, opType, profilingTitle, inputs),
      pgUID_(std::move(pgUID)),
      pgDesc_(std::move(pgDesc)),
      device_(device),
      workStartTime_(std::chrono::steady_clock::now()),
      seq_(seq),
      isP2P_(isP2P),
      timingEnabled_(enableTiming),
      distDebugLevel_(distDebugLevel) {
  // Creates the MUSA event wrappers
  // Note: The actual events are lazily created when first recorded to with
  // DEFAULT_FLAGS = cudaEventDisableTiming.
  if (musaEventCacheEnabled) {
    mcclStartEvent_ = enableTiming
        ? ProcessGroupMCCL::MUSAEventCache::get(device.index())
              ->create(enableTiming)
        : nullptr;
    mcclEndEvent_ = ProcessGroupMCCL::MUSAEventCache::get(device.index())
                        ->create(enableTiming);
  } else {
    mcclStartEvent_ = enableTiming
        ? std::make_shared<at::musa::MUSAEvent>(musaEventDefault)
        : nullptr;
    mcclEndEvent_ = std::make_shared<at::musa::MUSAEvent>(
        enableTiming ? musaEventDefault : musaEventDisableTiming);
  }
  futureWorkResult_ =
      c10::make_intrusive<at::ivalue::Future>(c10::AnyEnumType::get());
}

ProcessGroupMCCL::WorkMCCL::WorkMCCL(const WorkMCCL& w)
    : Work(w.rank_, w.opType_),
      std::enable_shared_from_this<WorkMCCL>(w),
      pgUID_(w.pgUID_),
      pgDesc_(w.pgDesc_),
      device_(w.device_),
      mcclStartEvent_(w.mcclStartEvent_),
      mcclEndEvent_(w.mcclEndEvent_),
      mcclComm_(w.mcclComm_),
      blockingWait_(w.blockingWait_),
      opTimeout_(w.opTimeout_),
      ownedEphermeralTimeout_(w.ownedEphermeralTimeout_),
      workStartTime_(w.workStartTime_),
      seq_(w.seq_),
      isP2P_(w.isP2P_),
      startTraceUpdated_(w.startTraceUpdated_),
      numelIn_(w.numelIn_),
      numelOut_(w.numelOut_),
      store_(w.store_),
      futureWorkResult_(w.futureWorkResult_),
      timingEnabled_(w.timingEnabled_),
      trace_id_(w.trace_id_),
      distDebugLevel_(w.distDebugLevel_) {
  exception_ = w.exception_;
}

bool ProcessGroupMCCL::WorkMCCL::isCompleted() {
  checkAndSetException();
  return exception() || finishedGPUExecutionInternal();
}

bool ProcessGroupMCCL::WorkMCCL::isStarted() {
  checkAndSetException();
  return exception() || startedGPUExecutionInternal();
}

bool ProcessGroupMCCL::WorkMCCL::isSuccess() const {
  C10_THROW_ERROR(NotImplementedError, "WorkMCCL::isSuccess() is deprecated");
}

void ProcessGroupMCCL::WorkMCCL::checkAndSetException() {
  if (exception()) {
    // We already have an exception.
    return;
  }

  auto exception_ptr = checkForMCCLErrors();
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
  if (exception_) {
    LOG(ERROR) << logPrefix() << "Collective " << *this
               << " raised the following async exception: "
               << getExceptionMsgFromExceptionPtr(exception_);

    // Mark future result as ERROR
    if (futureWorkResult_ && !futureWorkResult_->completed()) {
      futureWorkResult_->markCompleted(
          at::IValue(static_cast<uint8_t>(WorkResult::COMM_ERROR)));
    }
  }
}

const std::string& ProcessGroupMCCL::WorkMCCL::logPrefix() const {
  static std::string prefix = c10::str("[Rank ", rank_, "] ");
  return prefix;
}

void ProcessGroupMCCL::WorkMCCL::setException(
    std::exception_ptr exception_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = std::move(exception_ptr);
}

// Helper that checks if the MCCL kernels are completed on the GPUs
bool ProcessGroupMCCL::WorkMCCL::finishedGPUExecution() {
  checkAndSetException();
  return finishedGPUExecutionInternal();
}

bool ProcessGroupMCCL::WorkMCCL::startedGPUExecutionInternal() const {
  // if timing is disabled we won't have allocated start events
  if (!timingEnabled_) {
    return false;
  }
  // Checking the work's corresponding MUSA event's status
  if (!mcclStartEvent_->query()) {
    return false;
  }
  return true;
}

bool ProcessGroupMCCL::WorkMCCL::finishedGPUExecutionInternal() const {
  // Checking the work's corresponding MUSA event's status
  // It calls `cudaEventQuery` eventually. Although this seems to be a
  // non-blocking call, but we did notice hangs in the past. It can
  // hang if another thread is holding the MUSA global context lock. For
  // example, when doing a `cudaDeviceSynchronize` or even
  // `cudaStreamSynchronize`.
  if (!mcclEndEvent_->query()) {
    return false;
  }
  return true;
}

bool ProcessGroupMCCL::WorkMCCL::checkTimeout(
    std::optional<std::chrono::milliseconds> timeout) {
  // STATIC_SCOPED_WAIT_COUNTER(
  //     pytorch.wait_counter.ProcessGroupMCCL__checkTimeout);
  auto currentTimepoint = std::chrono::steady_clock::now();
  auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      currentTimepoint - workStartTime_);
  auto workTimeout = timeout ? *timeout : opTimeout_;

  if (timeElapsed < workTimeout) {
    return false;
  }

  // Timed out

  std::string exceptionMsg = c10::str(
      logPrefix(),
      "Watchdog caught collective operation timeout: ",
      *this,
      " ran for ",
      timeElapsed.count(),
      " milliseconds before timing out.");

  LOG(ERROR) << exceptionMsg;

  std::exception_ptr exception_ptr =
      std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exceptionMsg));
  if (!exception()) {
    // if there is already an error, we don't override it
    setException(exception_ptr);
  }

  // Mark future result as TIMEOUT
  if (futureWorkResult_ && !futureWorkResult_->completed()) {
    futureWorkResult_->markCompleted(
        at::IValue(static_cast<uint8_t>(WorkResult::TIMEOUT)));
  }
  return true;
}

// Print the traceback of the collective at call time
void ProcessGroupMCCL::WorkMCCL::printTraceback() const {
  // First step we get the corresponding record entry from FR, based on work's
  // trace_id_
  std::optional<FlightRecorder::Entry> entry =
      FlightRecorder::get()->getEntry(trace_id_);
  if (entry.has_value()) {
    auto entryVal = entry.value();
    // Get stack trace from FR entry, in string format
    // Note: `getTraceback` call below invokes `torch::symbolize`, which may
    // need to acquire the GIL. In order for watchdog to be block-free, we make
    // the call with std::async.
    auto future = std::async(
        std::launch::async, [&entryVal]() { return entryVal.getTraceback(); });
    // Wait for the future to complete or timeout
    auto status = future.wait_for(std::chrono::seconds(8));
    if (status == std::future_status::ready) {
      std::string tracebackStr = future.get();
      LOG(ERROR) << "Stack trace of the failed collective: \n" << tracebackStr;
    } // else, symbolizer probably timed out, we skip logging the stack trace.
  } else {
    LOG(ERROR)
        << "Stack trace of the failed collective not found, "
        << "potentially because FlightRecorder is disabled. "
        << "You can enable it by setting TORCH_MCCL_TRACE_BUFFER_SIZE to a non-zero value.";
  }
}

void ProcessGroupMCCL::WorkMCCL::handleException(
    ErrorHandlingMode errorHandling) {
  if (exception_) {
    auto exceptionMsg = c10::str(
        "Some MCCL operations have failed or timed out. Due to the ",
        "asynchronous nature of MUSA kernels, subsequent GPU operations ",
        "might run on corrupted/incomplete data.");
    LOG(ERROR) << logPrefix() << exceptionMsg;
    C10_LOG_API_USAGE_ONCE("ProcessGroupMCCL.WorkMCCL.handleException");

    auto logger = c10d::C10dLogger::getLogger();
    if (logger) {
      ::c10d::C10dLoggingData data;
      data.strings["work_mccl_exception"] =
          getExceptionMsgFromExceptionPtr(exception_);
      logger->log(data);
    }

    if (SHOULD_TEAR_DOWN(errorHandling)) {
      auto tearDownMsg = c10::str(
          "To avoid data inconsistency, we are taking the entire process down.");
      LOG(ERROR) << logPrefix() << tearDownMsg;
      std::rethrow_exception(exception_);
    }
  }
}

void ProcessGroupMCCL::WorkMCCL::synchronize() {
  synchronizeStream();
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(
        c10::intrusive_ptr<
            ProcessGroupMCCL::WorkMCCL>::unsafe_reclaim_from_nonowning(this));
  }
}

void ProcessGroupMCCL::WorkMCCL::synchronizeStream() {
  auto currentStream = at::musa::getCurrentMUSAStream(device_.index());
  // Block the current stream on the MCCL stream
  mcclEndEvent_->block(currentStream);

  if (avoidRecordStreams_) {
    stashed_for_allocator_safety_->clear();
  }
}

// Same as calling synchronize() when blockingWait_ is false
bool ProcessGroupMCCL::WorkMCCL::wait(std::chrono::milliseconds timeout) {
  RECORD_PARAM_COMMS(
      std::make_tuple(static_cast<int64_t>(this->seq_), this->isP2P_), // seq
      std::make_tuple(pgUID_, pgDesc_), // PG name tuple
      rank_, // rank
      "wait", // colName
      0, // inSize
      0, // outSize
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1,
      -1,
      static_cast<int>(1)); // number of device?

  // synchronize() will block the current stream on the MCCL stream
  synchronize();

  // In case of blockingWait or a timeout value is specified by the user, we
  // block the CPU thread until the work is completed or timed out.
  if (blockingWait_ || timeout != kNoTimeout) {
    while (!isCompleted()) {
      bool timedOut = checkTimeout(
          timeout == kNoTimeout ? std::nullopt : std::make_optional(timeout));
      // Explicitly abort mcclComms here before throwing this timed out
      // exception to users.
      // If throwing timed out excepiton without aborting mccl communicators
      // here, it was observed that MUSA GPU will have 100% utilization and
      // can not run new events successfully.
      if (timedOut) {
        std::string exceptionMsg = c10::str(
            logPrefix(), "Work ", (*this), " timed out in blocking wait.");
        LOG(ERROR) << exceptionMsg;
        break;
      }
      // Yield
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
  } else if (isBarrierOp_ && !isCompleted()) {
    // For barrier wait when timeout is unspecified, we block the CPU thread on
    // current stream. This is to minimize the CPU barrier wait time in healthy
    // path
    auto currentStream = at::musa::getCurrentMUSAStream(device_.index());
    // CUDAStream wrapper will correctly use a DeviceGuard here
    currentStream.synchronize();
  }

  // If exception is detected, throw it from the main CPU thread
  if (exception()) {
    // Abort MCCL communicators
    abort();
    // Throw exception (from main thread here)
    handleException(TearDown);
  }
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupMCCL::WorkMCCL::abort() {
  // Abort all communicators of this work
  mcclComm_->abort();

  mcclCommDevIdxMapMutex.lock();
  mcclCommDevIdxMap.erase(mcclComm_);
  mcclCommDevIdxMapMutex.unlock();
}

ProcessGroupMCCL::MUSAEventCache::MUSAEventCache() = default;

// MUSA event is used to record the start/end of one Work.
// Instead of let the MUSA event gets destroyed, we now reuse it after the Work
// has been erased from workMetaList_.
// This is to avoid the potential deadlock caused by CudaEventDestroy.
std::shared_ptr<at::musa::MUSAEvent> ProcessGroupMCCL::MUSAEventCache::create(
    bool timing) {
  // Register the deleter as a callback when the WorkMCCL object is destroyed.
  // Each deleter keeps a ref count to the cache object, so that even when
  // the thread that creates the cache is gone, the cache object won't be
  // destroyed until all the events in the cache are destroyed (ref number drops
  // to zero).
  auto deleter = [cache = shared_from_this(),
                  timing](at::musa::MUSAEvent* event) {
    std::lock_guard<std::mutex> lock(cache->cacheMutex_);
    // We put the event back to the cache deque once the WorkMCCL object is
    // destroyed.
    cache->eventsArray_[timing ? 1 : 0].push_back(event);
  };
  at::musa::MUSAEvent* event = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    auto& events = eventsArray_[timing ? 1 : 0];
    // If we still have events in the cache, we reuse it. Otherwise, we create a
    // new one.
    if (!events.empty()) {
      event = events.front();
      events.pop_front();
    } else {
      event = new at::musa::MUSAEvent(
          timing ? musaEventDefault : musaEventDisableTiming);
    }
  }
  return std::shared_ptr<at::musa::MUSAEvent>(event, std::move(deleter));
}

std::shared_ptr<ProcessGroupMCCL::MUSAEventCache> ProcessGroupMCCL::
    MUSAEventCache::get(at::DeviceIndex device) {
  // A per-thread singleton of device-to-CUDAEventCache map.
  // Map is needed because events cannot be reused across devices.
  // Per-thread ownership is needed to support multi-threaded case (instead of
  // multi-process case).
  static thread_local std::
      map<at::DeviceIndex, std::shared_ptr<ProcessGroupMCCL::MUSAEventCache>>
          cacheDeviceMap;
  // Check if device has already been in the map, if not, add a new entry
  auto it = cacheDeviceMap.find(device);
  if (it == cacheDeviceMap.end()) {
    cacheDeviceMap.emplace(
        device, std::make_shared<ProcessGroupMCCL::MUSAEventCache>());
  }
  return cacheDeviceMap[device];
}

static std::atomic<size_t> process_group_id = 0;

constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple. You are probably using multiple "
    "devices under one thread. The support for such usage has been deprecated. "
    "For details, please refer to "
    "https://pytorch.org/docs/stable/distributed.html#multi-gpu-collective-functions. "
    "ProcessGroupMCCL continues supporting multi-process and multi-thread modes.";

ProcessGroupMCCL::ProcessGroupMCCL(
    c10::intrusive_ptr<Store> store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(std::move(store)),
      options_(std::move(options)),
      terminateProcessGroup_(false),
      terminateHeartbeatMonitorThread_(false),
      local_id_(process_group_id++),
      intraNodeComm_(initIntraNodeComm()) {
  TORCH_CHECK(
      c10::musa::device_count() != 0,
      "ProcessGroupMCCL is only supported with GPUs, no GPUs found!");

  this->setGroupUid(options_->group_name);
  this->localDeviceCount_ = static_cast<int>(c10::musa::device_count());
  logPrefix_ = createLogPrefix();
  blockingWait_ = getCvarBool(TORCH_MCCL_BLOCKING_WAIT, false);
  asyncErrorHandling_ = static_cast<ErrorHandlingMode>(
      getCvarInt(TORCH_MCCL_ASYNC_ERROR_HANDLING, 3 /*SkipCleanUp*/));
  desyncDebug_ = getCvarBool(TORCH_MCCL_DESYNC_DEBUG, false) ||
      (dist_debug_level_ >= DebugLevel::Detail);
  rethrowMUSAErrors_ = getCvarBool(TORCH_MCCL_RETHROW_MUSA_ERRORS, true);
  dumpOnTimeoutOrEx_ = getCvarBool(TORCH_MCCL_DUMP_ON_TIMEOUT, false) ||
      (dist_debug_level_ >= DebugLevel::Detail);
  propagatePgError_ = getCvarBool(TORCH_MCCL_PROPAGATE_ERROR, false);
  // logging C++ stack isn't safe. Introduce a variable to control it.
  logCppStackOnUncleanShutdown_ =
      getCvarBool(TORCH_MCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN, true);
  enableNanCheck_ = getCvarBool(TORCH_MCCL_NAN_CHECK, false);
  heartbeat_ = 1ULL;

  monitorThreadEnabled_.store(getCvarBool(TORCH_MCCL_ENABLE_MONITORING, true));
  musaEventCacheEnabled_.store(getCvarBool(TORCH_MCCL_MUSA_EVENT_CACHE, true));
  heartbeatTimeoutInSec_ =
      getCvarInt(TORCH_MCCL_HEARTBEAT_TIMEOUT_SEC, 60 * 8 /*8 Mins*/);
  waitTimeoutDumpInMilSec_ =
      getCvarInt(TORCH_MCCL_WAIT_TIMEOUT_DUMP_MILSEC, 60 * 1000 /*60 Sec*/);
  coordCheckIntervalMilSec_ = getCvarInt(TORCH_MCCL_COORD_CHECK_MILSEC, 1000);
  traceBufferSize_ = getCvarInt(TORCH_MCCL_TRACE_BUFFER_SIZE, 2000);
  enableCollecticeHashDebug_ = (dist_debug_level_ >= DebugLevel::Detail);

  // store_ usually is wrapped with PrefixStore and the prefix is different
  // across different ProcessGroupMCCL(PG) instances. We need to get the
  // underlying non-PrefixStore for sharing global information shared across
  // different PGs.
  PrefixStore* prefixStore = dynamic_cast<PrefixStore*>(store_.get());
  globalStore_ =
      prefixStore ? prefixStore->getUnderlyingNonPrefixStore() : store_;

  enableTiming_.store(
      getCvarBool(TORCH_MCCL_ENABLE_TIMING, false) || desyncDebug_);

  avoidRecordStreams_ = getCvarBool(TORCH_MCCL_AVOID_RECORD_STREAMS, false);

  if (blockingWait_) {
    LOG(INFO)
        << logPrefix()
        << "TORCH_MCCL_BLOCKING_WAIT is enabled, NO watchdog thread is created.";
  } else {
    if (desyncDebug_ && asyncErrorHandling_ == NoHandling) {
      LOG(INFO)
          << logPrefix()
          << "TORCH_MCCL_DESYNC_DEBUG and TORCH_MCCL_ASYNC_ERROR_HANDLING "
          << "must both be enabled. "
          << "Enabling TORCH_MCCL_ASYNC_ERROR_HANDLING.";
      asyncErrorHandling_ = SkipCleanUp;
    }
  }

  // in blockingWait mode, we don't need to enable the watchdog thread to check
  // the timeout or mccl error because the main thread would throw an exception
  // and it is the user's responsibility to handle the exception.
  if (!blockingWait_) {
    mcclCommWatchdogThread_ =
        std::thread(&ProcessGroupMCCL::mcclCommWatchdog, this);
  }

  init();
  const std::string OFF = "OFF";
  std::string torch_distributed_debug =
      getCvarString({"TORCH_DISTRIBUTED_DEBUG"}, OFF.c_str());
  LOG(INFO) << logPrefix() << "ProcessGroupMCCL initialization options: "
            << "size: " << size << ", global rank: " << globalRank()
            << ", TIMEOUT(ms): " << options_->timeout.count()
            << ", USE_HIGH_PRIORITY_STREAM: "
            << options_->is_high_priority_stream
            << ", SPLIT_FROM: " << options_->split_from
            << ", SPLIT_COLOR: " << options_->split_color
            << ", PG Name: " << options_->group_name;

  getGlobalRankStartAndStride(
      options_->global_ranks_in_group,
      this->globalRankStart,
      this->globalRankStride);

  // Attach hooks to cache allocator to trigger the hooks whenever a traced
  // action is called. In the following hooks, we register a newly allocated
  // segment when SEGMENT_ALLOC action occurs, and deregister a segment when
  // SEGMENT_FREE action occurs.
  // We attach hooks only once at the first PG creation.
  // Attaching hooks fails if CUDACachingAllocator is not initialized, so
  // Init for MUSA is called (and is a no-op if MUSA is already
  // initialized).
  if (useTensorRegisterAllocatorHook_ && !allocatorHooksAttached) {
    at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);
    c10::musa::MUSACachingAllocator::attachAllocatorTraceTracker(
        &cacheAllocatorRegisterHook);
    c10::musa::MUSACachingAllocator::attachAllocatorTraceTracker(
        &cacheAllocatorDeregisterHook);
    allocatorHooksAttached = true;
  }

  // Enable Desync Debugger per user setting
  if (desyncDebug_) {
    desyncDebugger_.init(rank, size, store_);
  }
}

void ProcessGroupMCCL::eagerConnectSingleDevice(at::Device device) {
  const auto key = getKeyFromDevice(device);
  LOG(INFO) << logPrefix() << "Eagerly connecting mccl backend with device "
            << device;
  initMCCLComm(key, device, OpType::ALLREDUCE);
}

// TODO: update it
std::string ProcessGroupMCCL::dump_mccl_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  auto mcclDumpMap = getMCCLCommDumpMap();
  return MCCLTraceBuffer::get()->dump(
      mcclDumpMap, includeCollectives, includeStackTraces, onlyActive);
}

// TODO: add non blocking mode
bool ProcessGroupMCCL::useNonblocking() {
  return false;
}

void ProcessGroupMCCL::performNocolorSplit(at::Device device) {
  // If our backend doesn't support splitting, this is a no-op for
  // ranks not in the new subgroup (and ranks that would be in it will
  // just use a new communicator rather than split).

  // MCCL has no comm split
}

bool ProcessGroupMCCL::isInitialized() {
  if (devMCCLCommMap_.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  bool initialized = true;
  for (const auto& [_, comm] : devMCCLCommMap_) {
    if (!comm->isInitialized()) {
      initialized = false;
      break;
    }
  }
  return initialized;
}

ErrorType ProcessGroupMCCL::getError() {
  std::lock_guard<std::mutex> lock(errorMutex_);
  return error_;
}

int ProcessGroupMCCL::initIntraNodeComm() {
  return 0;
}

void ProcessGroupMCCL::setSequenceNumberForGroup() {
} // MCCL just starts sequence numbers at 0.

uint64_t ProcessGroupMCCL::getSequenceNumberForGroup() {
  return seqCollective_;
}

void ProcessGroupMCCL::enableCollectivesTiming() {
  enableTiming_.store(true);
}

bool ProcessGroupMCCL::waitForFutureOrTimeout(
    std::future<bool>& fut,
    const std::chrono::milliseconds& timeOutMilSec,
    const std::string& futDescription,
    ::c10d::C10dLoggingData& debugLog,
    bool throwException) {
  std::string errorMsg;
  bool complete = false;

  TORCH_CHECK(fut.valid(), "Expected a valid future");
  std::future_status status = fut.wait_for(timeOutMilSec);
  if (status == std::future_status::ready) {
    // Calling .get() will re-raise any exception from the future, and we don't
    // care about the retval
    try {
      bool result = fut.get();
      if (result) {
        LOG(INFO) << logPrefix()
                  << "future is successfully executed for: " << futDescription;
        debugLog.strings["status"] = "SUCCESS";
        complete = true;
      }
    } catch (const std::exception& e) {
      errorMsg = c10::str(
          logPrefix(),
          "Exception thrown when waiting for future ",
          futDescription,
          ": ",
          e.what());

      debugLog.strings["status"] = "EXCEPTION";
      debugLog.strings["exception"] = e.what();
      LOG(ERROR) << errorMsg;
    } catch (...) {
      errorMsg = c10::str(
          logPrefix(),
          "Unknown exception thrown when waiting for future ",
          futDescription);
      debugLog.strings["status"] = "EXCEPTION";
      debugLog.strings["exception"] = "Unknown exception";
      LOG(ERROR) << errorMsg;
    }
  } else {
    errorMsg = c10::str(
        logPrefix(),
        "Future for ",
        futDescription,
        " timed out after ",
        timeOutMilSec.count(),
        " ms");
    LOG(ERROR) << errorMsg;
  }
  if (throwException && !errorMsg.empty()) {
    C10_THROW_ERROR(DistBackendError, errorMsg);
  }
  return complete;
}

void ProcessGroupMCCL::abortCommsFromMap(
    std::unordered_map<std::string, std::shared_ptr<MCCLComm>>& mcclCommsMap,
    const std::optional<std::string>& abortReason) {
  // The process may control multiple devices, loop through the communicators on
  // each device
  for (auto& it : mcclCommsMap) {
    auto& devName = it.first;
    auto& mcclComm = it.second;
    VLOG(2) << logPrefix() << "ProcessGroupMCCL destroying mcclComm_ "
            << mcclComm->repr() << " on MUSA device: " << devName;
    // abort() call now has GPU guard inside
    mcclComm->abort(abortReason);
    // Note that we don't remove the aborted communicators from the
    // cache. The reason is that if we do remove the communicator
    // from the cache, it is possible that a new collective operation
    // calls `mcclCommInitRank` to create a new communicator whereas
    // other ranks might have failed/timed out and didn't enter
    // `mcclCommInitRank`. As a result, when there is a failure on
    // a communicator the application receives an exception and its
    // their responsibility to destroy the process group and recreate
    // it to recover from errors.

    VLOG(2) << logPrefix() << "ProcessGroupMCCL destroyed "
            << " communicator on MUSA device: " << devName;
  }
}

// Abort all communicators on this rank
// Note: original name of this method is `abort`. It was renamed to
// `abortComms` to distinguish from the `abort` method below. The `abort`
// method calls `abortComms` but does more destruction than the latter.
bool ProcessGroupMCCL::abortComms(
    const std::optional<std::string>& abortReason) {
  // Remove record from global mcclCommDevIdxMapMutex before aboarting,
  // so that a new cache segment would not register to already aborded
  // communicators. Note that mcclCommDevIdxMap is a global container which may
  // contain other PG's communicators, thus we need to only erase communicators
  // for the current PG.
  mcclCommDevIdxMapMutex.lock();
  for (auto& it : devMCCLCommMap_) {
    auto& mcclComm = it.second;
    mcclCommDevIdxMap.erase(mcclComm);
  }
  mcclCommDevIdxMapMutex.unlock();

  std::lock_guard<std::mutex> lock(mutex_);
  abortCommsFromMap(devMCCLCommMap_, abortReason);
  abortCommsFromMap(inInitializationCommMap_, abortReason);
  return true;
}

// Abort this backend.
void ProcessGroupMCCL::abort() {
  // This will log counter for how long the abort actually takes.
  // STATIC_SCOPED_WAIT_COUNTER(pytorch.ProcessGroupMCCL__abort);

  // Don't join threads here since the purpose of this method is to abort all
  // communicators and signal the threads to exit. Joining on the threads could
  // potentially block and hence avoid it in this method.
  terminateProcessGroup_.store(true);
  workMetaListCV_.notify_one();

  // lauch abort asynchrounously and wait for it to complete or timeout
  LOG(INFO) << logPrefix()
            << "Launching ProcessGroupMCCL abort asynchrounously.";
  std::future<bool> fut =
      std::async(std::launch::async, [this]() { return this->abortComms(); });

  ::c10d::C10dLoggingData debugLog;
  waitForFutureOrTimeout(
      fut, options_->timeout, "ProcessGroup abort", debugLog, true);
  LOG(INFO) << logPrefix() << "ProcessGroupMCCL aborts successfully.";

  // We need to wait for abort to finish before we can safely shut down
  // heartbeat monitoring thread.
  terminateHeartbeatMonitorThread_.store(true);
  monitorWakeUpCV_.notify_one();
}

// Difference between `abort()` and `shutdown()`:
// 1. `abort()` will signal communicators to terminate all MCCL kernels
// immediately.
// 2. `shutdown()` will wait for all MCCL kernels to finish before destroying
// communicators.

// Destroy (shutdown) this backend -- normal exit.
void ProcessGroupMCCL::shutdown() {
  LOG(INFO) << logPrefix()
            << "Starting to destroy process group, flushing operations.";
  // Flush all collectives
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& it : devMCCLCommMap_) {
      auto& mcclComm = it.second;
      mcclComm->finalize();
    }
  }
  // Wait for all operations to complete.  If MCCL comm is non-blocking and
  // timeout is reach, this will throw an exception.
  for (auto& it : devMCCLCommMap_) {
    auto& mcclComm = it.second;
    // Use long interval to avoid acquiring CPU too frequently
    mcclComm->waitReady(true);
  }
  // // Deregister memory pool after finalizing all collectives
  // if (memPool_) {
  //   try {
  //     deregisterMemPool(memPool_.get());
  //   } catch (...) {
  //     LOG(ERROR) << logPrefix() << "Failed to deregister memory pool,
  //     ignoring";
  //   }
  // }
  // Tell watchdog to (1) flush its queue and (2) do not use comm objects
  // anymore because I am going to destroy them now
  LOG(INFO) << logPrefix() << "Operations flushed, joining watchdog thread.";
  terminateProcessGroup_.store(true);
  workMetaListCV_.notify_one();
  if (mcclCommWatchdogThread_.joinable()) {
    mcclCommWatchdogThread_.join();
  }
  // if (onCompletionHookThread_.joinable()) {
  //   onCompletionHookThread_.join();
  // }
  // Watchdog thread exiting, retire heartbeat monitoring thread now to avoid
  // false alarm
  terminateHeartbeatMonitorThread_.store(true);
  monitorWakeUpCV_.notify_one();
  // Destroy the communicator, reclaim resources
  LOG(INFO) << logPrefix() << "Watchdog joined, destroying MCCL communicators.";
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& it : devMCCLCommMap_) {
      auto& mcclComm = it.second;
      mcclComm->destroy();
    }
  }
  LOG(INFO) << logPrefix() << "Destroy complete.";
}

// NOLINTNEXTLINE(bugprone-exception-escape)
ProcessGroupMCCL::~ProcessGroupMCCL() {
  LOG(INFO) << logPrefix() << "ProcessGroupMCCL destructor entered.";

  // `shutdown()` or `abort` already called. Skip the favor of disposing
  // communicators.
  if (!terminateProcessGroup_.load()) {
    // If user haven't explicitly destroy/shutdown process group, destructor
    // needs to do so
    // First print warning on first rank of each node
    if (rank_ % localDeviceCount_ == 0) {
      TORCH_WARN_ONCE(
          "WARNING: destroy_process_group() was not called before program exit, "
          "which can leak resources. For more info, please see "
          "https://pytorch.org/docs/stable/distributed.html#shutdown");
    }

    // Note 1: in distributed_c10d.py, a reference to PG is held by the global
    // context. Therefore, we are here only when the global context is tearing
    // down, which means the entire program is exiting.  At this point, user
    // will no longer care about the result of any collective, thus we can use
    // abort instead of destroy to make the destruction non-blocking.

    // TODO: Note 1 is not true in case of a C++ program using libtorch, which
    // does not have the global context mentioned. In that case, calling
    // `abort()` here could lead to corrupted result. We should consider not
    // doing anything and just let things leak. Adversarial example:
    /*
      Work routine(Tensor& t) {
        pg = ProcessGroupMCCL(…);
        w = pg.allReduce(t);
        return w;
      }
    */
    abort();
  }

  // Make sure we've told threads to stop; doesn't hurt if we'd done so before.
  // Tell watchdog and onCompletionHook:
  terminateProcessGroup_.store(true);
  workMetaListCV_.notify_one();
  // Tell heartbeat thread:
  terminateHeartbeatMonitorThread_.store(true);
  monitorWakeUpCV_.notify_one();

  // Wait for all threads to finish before returning
  if (mcclCommWatchdogThread_.joinable()) {
    mcclCommWatchdogThread_.join();
    LOG(INFO) << logPrefix() << "ProcessGroupMCCL watchdog thread joined.";
  }
  if (mcclHeartbeatMonitorThread_.joinable()) {
    mcclHeartbeatMonitorThread_.join();
    LOG(INFO) << logPrefix()
              << "ProcessGroupMCCL heart beat monitor thread joined.";
  }
  // if (onCompletionHookThread_.joinable()) {
  //   onCompletionHookThread_.join();
  //   LOG(INFO) << logPrefix()
  //             << "ProcessGroupMCCL onCompletionHookThread thread joined.";
  // }
}

bool ProcessGroupMCCL::dumpDebuggingInfo(bool includeStackTrace /*=true*/) {
  // Serialize all calls to this function to avoid corrupting data, but allow
  // multiple calls in one runtime. User is responsible for preserving the
  // output file from an earlier call before a later call overwrites it.
  static std::mutex writeDebugInfoMutex;
  std::lock_guard<std::mutex> lock(writeDebugInfoMutex);
  LOG(ERROR)
      << logPrefix()
      << "ProcessGroupMCCL preparing to dump debug info. Include stack trace: "
      << includeStackTrace;
  if (traceBufferSize_ > 0) {
    // We dump mccl trace into local disk by default and users can register
    // their customized writer by inheriting `DebugInfoWriter` via
    // `registerDebugInfoWriter`.
    auto mcclTrace = dump_mccl_trace(true, includeStackTrace, false);
    DebugInfoWriter& writer = DebugInfoWriter::getWriter(globalRank());
    LOG(INFO) << logPrefix() << "ProcessGroupMCCL dumping mccl trace to "
              << writer.getWriterTarget();
    writer.write(mcclTrace);
    return true;
  }
  return false;
}

void ProcessGroupMCCL::terminateProcess(const std::string& errMsg) {
  // Logging with `FATAL`, after errMsg printed, it calls `std::abort()`
  // to terminate the program execution.
  LOG(FATAL) << logPrefix() << errMsg;
}

static long computeDeltaMS(
    std::chrono::time_point<std::chrono::steady_clock> start,
    std::chrono::time_point<std::chrono::steady_clock> end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

std::string ProcessGroupMCCL::getMCCLWatchdogTimeoutErrorMsg(
    const std::string& extraMsg) {
  return c10::str(
      logPrefix(),
      "Received a dump signal due to a collective timeout from ",
      extraMsg,
      " and we will try our best to dump the debug info. ",
      "Last enqueued MCCL work: ",
      pgStatus_->lastEnqueuedSeq,
      ", last completed MCCL work: ",
      pgStatus_->lastCompletedSeq,
      ".",
      "This is most likely caused by incorrect usages of collectives, e.g., wrong ",
      "sizes used across ranks, the order of collectives is not same for all ranks ",
      "or the scheduled collective, for some reason, didn't run. Additionally, ",
      "this can be caused by GIL deadlock or other reasons such as network errors or ",
      "bugs in the communications library (e.g. MCCL), etc. ");
}

std::string ProcessGroupMCCL::getMCCLWatchdogTimeoutExitMsg(
    const std::string& exitReason) {
  return c10::str(
      logPrefix(),
      "Terminating the process after attempting to dump debug info, due to ",
      exitReason,
      ".");
}

void ProcessGroupMCCL::heartbeatMonitor() {
  // c10::setThreadName("pt_mccl_heartbt");

  uint64_t heartBeatCounter = 0ULL;
  std::string errorMsg;
  std::string exitReason;
  bool checkDumpSignal = (dumpOnTimeoutOrEx_ && local_id_ == 0);
  int monitorPollInterval = checkDumpSignal || propagatePgError_
      ? coordCheckIntervalMilSec_
      : heartbeatTimeoutInSec_ * 1000;
  auto lastTimePollStore = std::chrono::steady_clock::now();
  auto lastTimeHeartBeatCheck = std::chrono::steady_clock::now();
  std::optional<DumpPipe> dumpPipe = std::nullopt;
  if (local_id_ == 0) {
    // DumpPipe is one per-trainer process, and its convenient to name them
    // after 'global' ranks in the system, So we assume processgroup (uid)==0 is
    // the global PG and has globally unique rank ids across trainers.
    dumpPipe.emplace(rank_);
  }
  while (true) {
    // This won't have any lock since this lock is only used here.
    // Please be aware that mutex `monitorMutex_` should not be used
    // somewhere else to avoid the deadlock.
    std::unique_lock<std::mutex> lock(monitorMutex_);
    if (monitorWakeUpCV_.wait_for(
            lock, std::chrono::milliseconds(monitorPollInterval), [&] {
              return terminateHeartbeatMonitorThread_.load();
            })) {
      // For the normal complete or user interception, monitorWakeUpCV_
      // will get notified, we early return and exit heartbeatMonitor.
      return;
    }
    auto currentTime = std::chrono::steady_clock::now();

    if (propagatePgError_) {
      // Check and set remote error if it has not been set before
      checkAndSetRemoteError();
    }

    // We put extra functionality in the thread for the default PG (aka,
    // local_id_=0) because the signal is same across different PGs. We only
    // need to run once per process to avoid duplicate things performed in too
    // many separate threads. For example, we check a global flag on the
    // TCPStore periodically to see if any PG on any rank observed a timeout and
    // signaled peers to dump debugging info, and we avoid hammering the
    // TCPStore from all PGs on the same rank.
    if (checkDumpSignal) {
      // There are two scenarios where monitor thread will dump on timeout:
      // 1. The current rank is the first to observe a timeout in watchdog.
      // (shouldDump_ was set to true by the watchdog thread).
      // 2. Other ranks detected the timeout and signal the current rank to
      // dump. In addtion, monitor threads will dump if watchdog threads has no
      // heartbeat or dumpPipe is not empty.
      if (shouldDump_.load()) {
        errorMsg = getMCCLWatchdogTimeoutErrorMsg("this local rank");
        exitReason = "collective timeout or exception";
        break;
      }
      // We poll store to see if some ranks have flagged a timeout when
      // we haven't polled for `heartbeat_timeout` seconds and there haven't
      // any work added or removed for `watchdog_timeout` seconds.
      if (computeDeltaMS(lastWorkListUpdateTime_, currentTime) >=
              kWatchdogThreadSleepMillis &&
          computeDeltaMS(lastTimePollStore, currentTime) >=
              coordCheckIntervalMilSec_) {
        lastTimePollStore = currentTime;
        auto handleError = [&](const std::string& errorMessage) {
          LOG(WARNING)
              << logPrefix()
              << "Failed to check the \"should dump\" flag on TCPStore, "
              << "(maybe TCPStore server has shut down too early), with error: "
              << errorMessage;
          // We give up for now assuming TCPStore has been torn down.
          return;
        };
        // Wrap globalStore_->check() in a try-catch block to avoid crashing if
        // the store is not available.
        bool checkExceptionDump = false;
        try {
          checkExceptionDump =
              globalStore_->check({std::string(kStoreDumpKey)});
        } catch (const c10::DistNetworkError& e) {
          handleError(e.msg());
        } catch (const std::exception& e) {
          handleError(e.what());
        }

        if (checkExceptionDump) {
          int timeOutRank = -1;
          if (!shouldDump_.load()) {
            LOG(ERROR)
                << logPrefix()
                << "Observed flight recorder dump signal from another rank via TCPStore.";
          }
          shouldDump_.store(true);
          try {
            auto vec = globalStore_->get(std::string(kStoreDumpKey));
            TORCH_CHECK_WITH(
                DistBackendError,
                vec.size() == sizeof(int),
                "Invalid size for the timeout rank ID");
            std::memcpy(&timeOutRank, vec.data(), vec.size());
          } catch (const std::exception& e) {
            LOG(ERROR) << logPrefix()
                       << "Failed to get timeout rank ID from TCPStore."
                       << e.what();
          }
          errorMsg =
              getMCCLWatchdogTimeoutErrorMsg(c10::str(" rank ", timeOutRank));
          exitReason = "collective timeout or exception";
          break;
        }
      }
    }

    if (computeDeltaMS(lastTimeHeartBeatCheck, currentTime) >=
        heartbeatTimeoutInSec_ * 1000l) {
      // Check the heart beat of watchdog thread.
      lastTimeHeartBeatCheck = currentTime;
      auto heartbeat = heartbeat_.load();
      if (heartbeat != heartBeatCounter) {
        heartBeatCounter = heartbeat;
      } else {
        shouldDump_.store(true);
        // Watchdog heartbeat timeout.
        errorMsg = c10::str(
            logPrefix(),
            "ProcessGroupMCCL's watchdog got stuck for ",
            heartbeatTimeoutInSec_,
            " seconds without making progress in monitoring enqueued collectives. ",
            "This typically indicates a MCCL/MUSA API (e.g., MusaEventDestroy) hang blocking the watchdog, ",
            "and could be triggered by another thread holding the GIL inside a ",
            "MUSA api (for example, MusaEventDestroy), or other deadlock-prone behaviors.",
            "If you suspect the watchdog is not actually stuck and a longer timeout would help, ",
            "you can either increase the timeout (TORCH_MCCL_HEARTBEAT_TIMEOUT_SEC) to a larger value "
            "or disable the heartbeat monitor (TORCH_MCCL_ENABLE_MONITORING=0)."
            "If either of aforementioned helps, feel free to file an issue to PyTorch about the short timeout "
            "or false positive abort; otherwise, please attempt to debug the hang. ");
        exitReason = "ProcessGroupMCCL watchdog hang";
        break;
      }
    }

    // process a request to dump the trace. only PG uid 0 will respond to dump
    // requests, but this is fine since all PG's feed into the same flight
    // recorder and dump. After dump, the training should continue.
    if (dumpPipe.has_value() && dumpPipe->shouldDump()) {
      // best effort dump, not waiting for the dump here
      std::future<bool> fut = std::async(
          std::launch::async, [this]() { return this->dumpDebuggingInfo(); });
    }
  }
  LOG(ERROR) << errorMsg;

  // We perform some checks to help users debug the timeout/hang issue:
  // 1. Dump the mccl trace (flight recorder) to help debug the issue
  //    (timeout after waitTimeoutDumpInMilSec_, which is one minute).
  // 2. Check if there is a GIL deadlock (timeout after 300ms).
  // 3. Try to dump the c++ stacktraces (blocking and would hang,
  //    users can turn this off by set
  //    TORCH_MCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN=0).

  // Dump the mccl trace (flight recorder).
  if (checkDumpSignal && shouldDump_.load()) {
    // Store debug info to storage if no other thread does it. (By default to
    // local disk)
    bool dumpStackTrace = true;
    ::c10d::C10dLoggingData debugLog;
    debugLog.integers["pg_id"] = static_cast<int64_t>(local_id_);
    debugLog.integers["rank"] = rank_;
    debugLog.integers["global_rank"] = globalRank();
    debugLog.integers["world_size"] = getSize();
    for (int i = 0; i < 2; i++) {
      std::future<bool> asyncDebugDump =
          std::async(std::launch::async, [this, dumpStackTrace]() {
            return this->dumpDebuggingInfo(dumpStackTrace);
          });

      // wait for the dump until timeout - log data
      auto complete = waitForFutureOrTimeout(
          asyncDebugDump,
          std::chrono::milliseconds(waitTimeoutDumpInMilSec_),
          "Flight recorder dump in heartbeatMonitor",
          debugLog,
          false);

      if (complete) {
        LOG(INFO)
            << logPrefix()
            << "Finished flight recorder successfully. Output can be analyzed using the fr_trace script.";
        break;
      }
      // If we failed to dump, try dumping without stack trace in the 2nd
      // iteration.
      dumpStackTrace = false;
    }
    debugLog.integers["trace_enabled"] = int64_t(dumpStackTrace);
    auto logger = c10d::C10dLogger::getLogger();
    if (logger) {
      logger->log(debugLog);
    }
    // Indicate to watchdog thread that we have finished dumping.
    promiseFlightRecorderDump_.set_value();
  }

  // GIL deadlock check.
  if (get_gil_checker() != nullptr) {
    auto fut = launchAsyncGilCheck();
    auto kGilCheckTimeout = std::chrono::milliseconds(300);
    auto futStatus = fut.wait_for(kGilCheckTimeout);
    if (futStatus != std::future_status::ready) {
      TORCH_CHECK(
          futStatus != std::future_status::deferred,
          "Expected the future to have been launched eagerly.");
      LOG(ERROR)
          << logPrefix()
          << "Could not acquire GIL within 300 ms on exit, possible GIL induced hang";
    }
  } else {
    VLOG(2)
        << logPrefix()
        << "GIL checker was not registered, perhaps this is a no-python build?";
  }

  // Dump the c++ stacktraces.
  auto& cpp_dumper = get_cpp_trace_dumper();
  if (logCppStackOnUncleanShutdown_ && cpp_dumper.has_value()) {
    LOG(INFO) << logPrefix() << "Dumping c++ stacktraces:";
    cpp_dumper.value()(
        [&](const std::string& line) { LOG(INFO) << logPrefix() << line; });
    LOG(INFO) << logPrefix() << "Finished c++ stacktraces dump.";
  }

  // There are two possible cases for the watchdog thread exit:
  // Case one: desync report runs quickly, and it follows the step:
  // collective timeout -> desync -> exception handling -> destructors
  // -> set terminateHeartbeatMonitorThread_ -> notify monitorWakeUpCV_.
  // So the code either early returns above or will skip the sleep below.
  // Case two: desync might be slow or get stuck. Or we get stuck in
  // destructors, we will sleep for some time before calling std::abort() to
  // kill the whole process.
  if ((terminateProcessGroup_.load() || desyncDebug_ || shouldDump_.load()) &&
      !terminateHeartbeatMonitorThread_.load()) {
    // Leave another two mins for desync report generation or process group
    // destroy.
    std::this_thread::sleep_for(std::chrono::seconds(heartbeatTimeoutInSec_));
    LOG(INFO) << logPrefix() << "slept for " << heartbeatTimeoutInSec_
              << " waiting for desync report or process group destroy.";
  }

  // At this point, we either already sleep for another `heartbeatTimeoutInSec_`
  // or the thread has finished. Because we don't want to block the monitor
  // thread, so We mark the thread detach and the dump of debug info becomes
  // "best effort". If the process exit normally, marking it detach also makes
  // sense because we don't really care about dumping the debug info.

  // We already log completion inside the thread, so it may not be necessary to
  // check the return value here.  We mainly use a future so we can exit early
  // if done.

  if (!terminateHeartbeatMonitorThread_.load()) {
    // Create a error message reported from MonitorThread, so
    // we throw exception and make the whole process to be killed.
    // TODO(fduwjj): After having a hang debug wiki, we need to update the wiki
    // url here.
    if (monitorThreadEnabled_.load()) {
      terminateProcess(getMCCLWatchdogTimeoutExitMsg(exitReason));
    } else {
      // Ideally we want to merge this one with the above one, but we are going
      // to remove the kill switch for monitor thread soon, so we keep this one
      // for now.
      LOG(ERROR)
          << logPrefix()
          << "ProcessGroupMCCL monitor thread is disabled, but would have terminated the process"
          << "after attempting to dump debug info, due to " << exitReason
          << ".";
    }
  }
}

void ProcessGroupMCCL::mcclCommWatchdog() {
  c10::setThreadName("pt_mccl_watchdg");

  try {
    VLOG(2) << logPrefix() << "Process group watchdog thread started!";
    mcclHeartbeatMonitorThread_ =
        std::thread(&ProcessGroupMCCL::heartbeatMonitor, this);
    watchdogHandler();
    VLOG(2) << logPrefix()
            << "Process group watchdog thread terminated normally";
  } catch (std::exception& e) {
    if (std::string(e.what()).find("driver shutting down") !=
        std::string::npos) {
      VLOG(2)
          << logPrefix()
          << "main process destroyed musa before watchdog loop exited, terminating watchdog."
          << " (Watchdog caught exception: " << e.what();

    } else {
      // Append error message reported from watchdogHandler
      const auto exitMsg = c10::str(
          logPrefix(),
          "Process group watchdog thread terminated with exception: ",
          e.what());
      LOG(ERROR) << exitMsg;
      if (C10_LIKELY(rethrowMUSAErrors_) ||
          !(std::string(e.what()).find("MUSA Error"))) {
        // TODO(whc) clean up the rethrow - why is it stored in a class var and
        // rethrown?
        watchDogException_ =
            std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
        std::rethrow_exception(watchDogException_);
      }
    }
  } catch (...) {
    const auto exitMsg = c10::str(
        logPrefix(),
        "Process group watchdog thread terminated with exception: unknown");
    LOG(ERROR) << exitMsg;
    watchDogException_ =
        std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
    std::rethrow_exception(watchDogException_);
  }
}

// Initialize and enable DesyncDebugger
void ProcessGroupMCCL::DesyncDebugger::init(
    int rank,
    int size,
    c10::intrusive_ptr<Store> store) {
  rank_ = rank;
  size_ = size;
  store_ = std::move(store);
  enabled_ = true;
  traceKeyStart_ = getTraceStartKey("MCCL", rank);
  traceKeyEnd_ = getTraceEndKey("MCCL", rank);
}

// Run desync debug. This function is called by watchdog at time of timeout.
void ProcessGroupMCCL::DesyncDebugger::run() {
  if (!enabled_)
    return;
  auto logPrefix = c10::str("Rank ", rank_);
  try {
    std::string desyncMsg = retrieveDesyncReport(store_, "MCCL", rank_, size_);
    LOG(ERROR) << logPrefix << desyncMsg;
  } catch (const std::exception& e) {
    enabled_ = false;
    LOG(ERROR) << logPrefix
               << " Failed to retrieve TORCH_MCCL_DESYNC_DEBUG report. "
               << " Please file an issue. Error: " << e.what();
  } catch (...) {
    enabled_ = false;
    LOG(ERROR)
        << logPrefix
        << " Failed to rerieve TORCH_MCCL_DESYNC_DEBUG report with unknown error."
        << " Please file an issue.";
  }
}

// Log work start to store.
void ProcessGroupMCCL::DesyncDebugger::logWorkStart(WorkMCCL& work) {
  if (!enabled_)
    return;
  if (work.startTraceUpdated_)
    return;

  work.startTraceUpdated_ = true;
  // If not successful, disable the debugger
  enabled_ = c10d::traceUpdate(
      store_, traceKeyStart_, work.seq_, opTypeToString(work.opType_));
}

// Log work end to store.
void ProcessGroupMCCL::DesyncDebugger::logWorkEnd(WorkMCCL& work) {
  if (!enabled_)
    return;

  // In case the start of the work hasn't been logged
  if (!work.startTraceUpdated_) {
    logWorkStart(work);
  }

  // If not successful, disable the debugger
  enabled_ = c10d::traceUpdate(
      store_, traceKeyEnd_, work.seq_, opTypeToString(work.opType_));
}

// We want to have both PG ID and global unique ID (guid) for the logging
// prefix. PG ID records how many ProcessGroupMCCL objects were created on a
// specific rank and is a stable index across ranks, which lets users reason
// about, for example, the second PG we initialized on this rank is for FSDP,
// and corresponds with PG ID = 1 on other ranks as well. Unlike PG ID, guid (or
// group name) is a global unique ID across ranks. The guid is either a hash of
// all the ranks in the group or a counter of how many times
// `_process_group_name` is called, essentially it means how many times we
// have PGs users have created. Before using split_group, even if
// we are creating a new sub-PG, all ranks have to call the API at the same
// time, and this makes `group_name` a unique identifier for a group (PG).
std::string ProcessGroupMCCL::createLogPrefix() const {
  if (!pg_desc_.empty() && pg_desc_ != "undefined") {
    return c10::str(
        "[PG ID ",
        local_id_,
        " PG GUID ",
        pg_uid_,
        "(",
        pg_desc_,
        ") Rank ",
        rank_,
        "] ");
  }
  return c10::str(
      "[PG ID ", local_id_, " PG GUID ", pg_uid_, " Rank ", rank_, "] ");
}

const std::string& ProcessGroupMCCL::logPrefix() const {
  return logPrefix_;
}

const int& ProcessGroupMCCL::globalRank() const {
  static int globalRank = rank_;
  return globalRank;
}

const std::vector<uint64_t>& ProcessGroupMCCL::groupRanks() const {
  if (options_->global_ranks_in_group.empty() && local_id_ == 0) {
    static std::vector<uint64_t> globalRanks(size_);
    std::iota(globalRanks.begin(), globalRanks.end(), 0);
    return globalRanks;
  }
  return options_->global_ranks_in_group;
}

void ProcessGroupMCCL::addEphemeralTimeout(
    const std::chrono::milliseconds& timeout) {
  std::lock_guard<std::mutex> timeoutLock(mtxTimeoutExtension_);
  ephemeralTimeoutActive_ += timeout;
}

bool ProcessGroupMCCL::verifyWorkTimeoutForTest(
    const c10::intrusive_ptr<Work>& work,
    const std::chrono::milliseconds& timeout) {
  // Since collective returns a c10d::Work, we need to cast it to WorkMCCL.
  if (auto workMCCL = c10::dynamic_intrusive_pointer_cast<WorkMCCL>(work)) {
    // workMCCL is now a c10::intrusive_ptr<WorkMCCL>
    return workMCCL->opTimeout_ == timeout;
  }
  C10_THROW_ERROR(
      DistBackendError, "Non c10d::WorkMCCL object returned from collective");
}

void ProcessGroupMCCL::broadcastSignal(
    c10::intrusive_ptr<Store>& store,
    const std::string& signal,
    int srcRank) {
  try {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(&srcRank),
        reinterpret_cast<uint8_t*>(&srcRank) + sizeof(srcRank));
    store->set(signal, vec);
    LOG(INFO) << logPrefix() << "Broadcasting signal " << signal
              << " to other ranks via TCPStore.";
  } catch (const std::exception& e) {
    LOG(ERROR) << logPrefix() << "Failed to broadcast signal " << signal
               << " through TCPStore. Error: " << e.what();
  }
}

int ProcessGroupMCCL::getSignalSrcRank(
    c10::intrusive_ptr<Store>& store,
    const std::string& signal) {
  // This function is 'non blocking'. We first 'check' if the key exists in the
  // store, then read/get the value only if the key exists.
  int srcRank = -1;
  bool signalExists = false;
  try {
    signalExists = store->check({signal});
  } catch (const std::exception& e) {
    LOG(WARNING) << logPrefix() << "Failed to check the signal " << signal
                 << " on TCPStore, " << e.what();
  }
  if (!signalExists) {
    return srcRank;
  }

  // key exists, now read and parse the value (source rank)
  std::vector<uint8_t> vec;
  try {
    vec = store->get(std::string(signal));
  } catch (const std::exception& e) {
    LOG(ERROR) << logPrefix() << "Failed to get source rank of the signal "
               << signal << " from TCPStore." << e.what();
  }
  TORCH_CHECK_WITH(
      DistBackendError,
      vec.size() == sizeof(int),
      "Invalid size for the timeout rank ID");
  std::memcpy(&srcRank, vec.data(), vec.size());
  return srcRank;
}

void ProcessGroupMCCL::broadcastDumpSignal() {
  // broadcast dump signal to all other global ranks.
  broadcastSignal(globalStore_, std::string(kStoreDumpKey), globalRank());
  // signal the local rank to start dumping
  if (shouldDump_.load()) {
    // already signaled dump, skipping signal again and wait for the dump
    // future.
    return;
  }
  LOG(ERROR) << logPrefix() << "First PG on this rank to signal dumping.";
  // signal the monitor thread on PG0 to start dumping
  shouldDump_.store(true);
  // Give time for dumping before throwing exception
  auto start = std::chrono::steady_clock::now();
  auto status = promiseFlightRecorderDump_.get_future().wait_for(
      std::chrono::milliseconds(waitTimeoutDumpInMilSec_));
  if (status == std::future_status::timeout) {
    LOG(WARNING) << logPrefix() << "timed out after waiting for "
                 << waitTimeoutDumpInMilSec_ << "ms"
                 << " flight recorder dumps to finish.";
  } else if (status == std::future_status::ready) {
    auto end = std::chrono::steady_clock::now();
    LOG(INFO) << logPrefix() << "slept for " << computeDeltaMS(start, end)
              << "ms"
              << " giving time for flight recorder dumps to finish.";
  }
}

void ProcessGroupMCCL::checkAndSetRemoteError() {
  // if the error is already set, no need to check again
  if (getError() != ErrorType::SUCCESS) {
    return;
  }
  // key/signal to read from the tcpstore is a string and pg specific:
  // format is: remote_error:pg_uid
  int remoteErrorRank = getSignalSrcRank(
      store_, std::string(kStoreErrorSignalKey) + ':' + pg_uid_);
  if (remoteErrorRank != -1) {
    std::lock_guard<std::mutex> lock(errorMutex_);
    error_ = ErrorType::REMOTE_ERROR;
    LOG(ERROR) << c10::str(
        logPrefix(), " remote error detected from rank: ", remoteErrorRank);
  }
}

void ProcessGroupMCCL::watchdogHandler() {
  bool done = false;
  lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
  auto lastStatusUpdateTime = std::chrono::steady_clock::now();
  std::list<ProcessGroupMCCL::WorkMCCL> completedWorkList;

  while (!done || !terminateProcessGroup_.load()) {
    std::unique_lock<std::mutex> lock(workMetaListMutex_);
    // We busy-poll the work vector every kWatchdogThreadSleepMillis
    // milliseconds as long as the atomic is True.
    workMetaListCV_.wait_for(
        lock,
        std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool { return terminateProcessGroup_.load(); });
    // Bump up heart beat by one.
    heartbeat_++;

    auto logger = ::c10d::C10dLogger::getLogger();
    if (logger &&
        computeDeltaMS(
            lastStatusUpdateTime, std::chrono::steady_clock::now()) >=
            kWorkStatusUpdatePeriodMs) {
      ::c10d::C10dLoggingData data;
      // logging integers
      data.integers["pg_id"] = static_cast<int64_t>(local_id_);
      data.integers["rank"] = rank_;
      data.integers["global_rank"] = globalRank();
      data.integers["last_enqueued_work"] = pgStatus_->lastEnqueuedSeq;
      data.integers["last_started_work"] = pgStatus_->lastStartedSeq;
      data.integers["last_completed_work"] = pgStatus_->lastCompletedSeq;
      data.integers["last_enqueued_numel_in"] =
          static_cast<int64_t>(pgStatus_->lastEnqueuedNumelIn);
      data.integers["last_enqueued_numel_out"] =
          static_cast<int64_t>(pgStatus_->lastEnqueuedNumelOut);
      data.integers["last_completed_numel_in"] =
          static_cast<int64_t>(pgStatus_->lastCompletedNumelIn);
      data.integers["last_completed_numel_out"] =
          static_cast<int64_t>(pgStatus_->lastCompletedNumelOut);
      data.integers["last_started_numel_in"] =
          static_cast<int64_t>(pgStatus_->lastStartedNumelIn);
      data.integers["last_started_numel_out"] =
          static_cast<int64_t>(pgStatus_->lastStartedNumelOut);
      // logging strings
      data.strings["last_enqueued_work_name"] = pgStatus_->lastEnqueuedWorkName;
      data.strings["last_started_work_name"] = pgStatus_->lastStartedWorkName;
      data.strings["last_completed_work_name"] =
          pgStatus_->lastCompletedWorkName;
      data.strings["pg_name"] = pg_uid_;
      data.strings["pg_desc"] = pg_desc_;
      logger->log(data);
      lastStatusUpdateTime = std::chrono::steady_clock::now();
    }

    for (auto it = workMetaList_.begin(); it != workMetaList_.end();
         /* no increment */) {
      auto& work = *it;
      // When terminateProcessGroup_ is true, communicators have already been
      // aborted, So cannot check exception based on them. But watchdog needs to
      // finish the check for the works that have already been enqueued to
      // workMetaList_

      // check MCCL errors first
      if (!terminateProcessGroup_.load()) {
        work.checkAndSetException();
      }

      if (work.exception()) {
        // set the error to the first error found
        std::lock_guard<std::mutex> lock(errorMutex_);
        if (error_ == ErrorType::SUCCESS) {
          error_ = ErrorType::COMM_ERROR;
        }
      }

      // Then check if work has timed out
      // Skip if work has encountered an error
      bool timedout = !work.exception() && work.checkTimeout();

      // Report desync state in case of timeout (if TORCH_MCCL_DESYNC_DEBUG is
      // turned on; otherwise, run() is no-op)
      if (timedout) {
        std::lock_guard<std::mutex> lock(errorMutex_);
        if (error_ == ErrorType::SUCCESS) {
          error_ = ErrorType::TIMEOUT;
        }
        desyncDebugger_.run();
      }

      // If work hits an exception (either an error or timeout)
      if (work.exception()) {
        LOG(ERROR) << c10::str(
            logPrefix(),
            " failure detected by watchdog at work sequence id: ",
            work.seq_,
            " PG status: last enqueued work: ",
            pgStatus_->lastEnqueuedSeq,
            ", last completed work: ",
            pgStatus_->lastCompletedSeq);

        // Print the traceback of the collective at call time
        work.printTraceback();

        // broadcast remote error signal to all other ranks in this specific PG.
        // key/signal to write in the tcpstore is a string and pg specific:
        // format is: remote_error:pg_uid
        if (propagatePgError_) {
          broadcastSignal(
              store_, std::string(kStoreErrorSignalKey) + ':' + pg_uid_, rank_);
        }

        // try to notify other ranks via global TCPStore to dump the flight
        // recorder when a collective timeout or exception happens. Flight
        // recorder behavior is independent of desync Debug.
        if (dumpOnTimeoutOrEx_) {
          broadcastDumpSignal();
        }

        if (SHOULD_CLEAN_UP(asyncErrorHandling_)) {
          // Abort work and corresponding communicators
          work.abort();
          // PG level abort, which would abort all other communicators on this
          // rank
          abortComms();
        }
        // Throw exception
        work.handleException(asyncErrorHandling_);
      }

      // Work status logging for desync debug
      desyncDebugger_.logWorkStart(work);

      // a work could be started but not completed, so we should not update
      // lastStartedSeq and lastStartedOpName if the work state is checked
      // multiple times after the start
      if (pgStatus_->lastStartedSeq < static_cast<int64_t>(work.seq_) &&
          work.isStarted()) {
        pgStatus_->lastStartedSeq = static_cast<int64_t>(work.seq_);
        pgStatus_->lastStartedWorkName = opTypeToString(work.opType_);
        pgStatus_->lastStartedNumelIn = work.numelIn_;
        pgStatus_->lastStartedNumelOut = work.numelOut_;
      }

      // allow watchdog to do an event query on a side thread
      at::musa::MUSAGuard device_guard(work.mcclEndEvent_->device_index());
      // TODO(MUSA): unsupported API
      // at::musa::MUSAStreamCaptureModeGuard
      // g{musaStreamCaptureModeThreadLocal};

      // Clean up completed work
      if (work.isCompleted()) {
        // Work status logging for desync debug
        desyncDebugger_.logWorkEnd(work);

        if (work.futureWorkResult_ && work.finishedGPUExecutionInternal() &&
            !work.futureWorkResult_->completed()) {
          work.futureWorkResult_->markCompleted(
              at::IValue(static_cast<uint8_t>(WorkResult::SUCCESS)));
        }
        {
          // Reset the timeout and first work if the work is completed.
          std::lock_guard<std::mutex> timeoutLock(mtxTimeoutExtension_);
          if (work.ownedEphermeralTimeout_.count() > 0) {
            ephemeralTimeoutActive_ -= work.ownedEphermeralTimeout_;
            ephemeralTimeoutInflight_ -= work.ownedEphermeralTimeout_;
          }
        }
        pgStatus_->lastCompletedSeq = static_cast<int64_t>(work.seq_);
        pgStatus_->lastCompletedWorkName = opTypeToString(work.opType_);
        pgStatus_->lastCompletedNumelIn = work.numelIn_;
        pgStatus_->lastCompletedNumelOut = work.numelOut_;
        FlightRecorder::get()->retire_id(work.trace_id_, true);

        it = workMetaList_.erase(it);
        lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
      } else {
        // Increment the iterator if the current WorkMCCL object is not
        // completed.
        ++it;
      }
      // Increment heartbeat after each work processed,
      // in case processing is slowed down (but not hung) by musa api contention
      heartbeat_++;
    }
    done = workMetaList_.empty();
  }
}

std::exception_ptr ProcessGroupMCCL::WorkMCCL::checkForMCCLErrors() {
  return checkForMCCLErrorsInternal(mcclComm_);
}

std::exception_ptr ProcessGroupMCCL::checkForMCCLErrors(
    std::shared_ptr<MCCLComm>& mcclComm) {
  return checkForMCCLErrorsInternal(mcclComm);
}

std::exception_ptr ProcessGroupMCCL::checkForMCCLErrorsInternal(
    std::shared_ptr<MCCLComm>& mcclComm) {
  // Prioritize commFailureReason over checkForMcclError() result if
  // commFailureReason is set.
  auto commFailureReason = mcclComm->getMcclCommFailureReason();
  if (commFailureReason != std::nullopt) {
    return std::make_exception_ptr(std::runtime_error(c10::str(
        "MCCL communicator encountered error set by ProcessGroupMCCL: ",
        *commFailureReason)));
  }
  mcclResult_t mcclAsyncErr = mcclComm->checkForMcclError();
  // When nonblocking mode is enabled by TORCH_NCCL_USE_COMM_NONBLOCKING,
  // ncclInProgress could be returned when there are pending MCCL calls.
  // In this case, no exception should be thrown
  // ncclInProgress is defined only if NCCL_HAS_COMM_NONBLOCKING is defined
  // TODO: add non blocking mode
  if (mcclAsyncErr != mcclSuccess) {
    return std::make_exception_ptr(std::runtime_error(
        "MCCL error: " + mcclGetErrorWithVersion(mcclAsyncErr) + "\n" +
        getMcclErrorDetailStr(mcclAsyncErr)));
  }
  return nullptr;
}

void ProcessGroupMCCL::broadcastUniqueMCCLID(
    mcclUniqueId* mcclID,
    bool isSingleP2POp,
    const std::string& p2pKey,
    int p2pRank) {
  // For collective operations:
  // For every MCCL communicator that we create we need to broadcast
  // a unique ID from rank 0 to all other ranks. This broadcast is
  // done by rank 0 setting a key in the store and all other ranks
  // retrieving the contents of that key. A single process group
  // may create multiple MCCL communicators, so we use a sequence
  // number to differentiate between them.
  // For single point-to-point operations:
  // The sequence number will only be increased on 2 out of all the
  // processes in a Process Group. So all following collective
  // operations will see different sequence numbers which will cause
  // runtime errors. To avoid that, use the src:target pair instead
  // of sequence number for p2p communications.

  std::string storeKey;
  if (!isSingleP2POp) {
    storeKey = std::to_string(mcclCommCounter_++);
  } else {
    storeKey = p2pKey;
  }
  if (rank_ == 0 || (isSingleP2POp && p2pRank == 0)) {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(mcclID),
        reinterpret_cast<uint8_t*>(mcclID) + MCCL_UNIQUE_ID_BYTES);
    store_->set(storeKey, vec);
  } else {
    try {
      auto vec = store_->get(storeKey);
      TORCH_CHECK(vec.size() == MCCL_UNIQUE_ID_BYTES);
      std::memcpy(mcclID, vec.data(), vec.size());
    } catch (const std::exception& e) {
      std::string exceptionMsg = c10::str(
          "[",
          rank_,
          "] is setting up MCCL communicator and "
          "retrieving McclUniqueId from [0] via c10d key-value store by key '",
          storeKey,
          "', but store->get('",
          storeKey,
          "') got error: ");
      TORCH_CHECK(
          false,
          exceptionMsg + e.what() +
              ". This may indicate a possible application crash on rank 0 or a network set up issue.");
    } catch (...) {
      TORCH_CHECK(
          false,
          c10::str(
              "Unknown exception while [",
              rank_,
              "] is setting up MCCL communicator and "
              "retrieving McclUniqueId from [0] via c10d key-value store by key '",
              storeKey,
              "'",
              ". This may indicate a possible application crash on rank 0 or a network set up issue."));
    }
  }
}

std::shared_ptr<MCCLComm> ProcessGroupMCCL::initMCCLComm(
    const std::string& deviceKey,
    at::Device& device,
    OpType opType,
    int p2pRank,
    bool isSendRecvSelf) {
  // Sanity check
  if (deviceKey.empty()) {
    C10_THROW_ERROR(
        DistBackendError,
        "Not able to create/get the MCCL Communicator since "
        "the GPU devices are not known");
  }
  if (bound_device_id_) {
    if (*bound_device_id_ != device) {
      LOG(ERROR) << logPrefix() << "Tensor found on device " << device
                 << " but backend constrained to " << *bound_device_id_;
      C10_THROW_ERROR(
          DistBackendError,
          "Attempt to perform collective on tensor not on device passed to init_process_group");
    }
  }

  usedDeviceIdxs_.insert(device.index());

  // MCCL communicator not cached, create a new entry
  std::shared_ptr<MCCLComm> mcclComm;

  // Create the unique MCCL ID and broadcast it
  mcclUniqueId mcclID;

  // reset log prefix to include group_desc
  logPrefix_ = createLogPrefix();

  // For batch_isend_irecv, ncclGroupStart() would be called upfront
  bool batchP2P = mcclActiveGroupCounter_ > 0;
  bool singleP2POp = isP2POp(opType, batchP2P);

  // Get the device index
  auto deviceIndex = device.index();
  at::musa::OptionalMUSAGuard gpuGuard(device);

  // [Group Start/End Note] This is used to ensure that mccl communicator will
  // be created before communication primitives are called. Let's look at this
  // example: Using the batch_isend_irecv to send a tensor to a target process.
  // On the sender side, the corresponding underlying MCCL calls will look like
  //   ncclGroupStart() // This is in batch_isend_irecv
  //   ncclCommInitRank() // Inside NCCLComm::create
  //   ncclSend()
  //   ncclGroupEnd() // This is in batch_isend_irecv
  // With this pattern, the mccl communicator will be created in the last
  // ncclGroupEnd which means when ncclSend is processed, the passed
  // communicator argument is NULL which will lead to runtime error. So we need
  // to "close" all active mccl groups to ensure mccl communicator is actually
  // created before encountering any communication calls. This is why we need
  // the following for loop.
  for (const auto i : c10::irange(mcclActiveGroupCounter_)) {
    (void)i;
    // comms have not been initiated yet, so can only check in blocking-way
    C10D_MCCL_CHECK(mcclGroupEnd(), std::nullopt);
  }

  // GPU world size and GPU rank
  int numRanks = -1, rank = -1;

  if (!singleP2POp) {
    // Collective, all-to-all, or batch P2P
    numRanks = getSize();
    rank = getRank();
  } else if (isSendRecvSelf) {
    // Same process send and recv.
    numRanks = 1;
    rank = 0;
  } else {
    // For single point-to-point operation, there are only 2 processes
    // involved so the GPU rank is either 0 or 1.
    numRanks = 2;
    rank = p2pRank;
  }

  RECORD_PARAM_COMMS(
      std::make_tuple(0, false), // seq
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      rank, // rank
      "init", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      size_); // worldSize

  // TODO: add non blocking mode
  // bool useNb = useNonblocking();
  // options_->config.blocking = useNb ? 0 : 1;

  // TODO : mccl does not support ScalableInit
  bool useScalableInit = false;
  if (useScalableInit) {
  } else {
    // To simplify conditional nesting, just create the ncclComms[i]
    // entry if it hasn't been yet rather than untangling the
    // conditions that might have resulted in a split above.
    if (!mcclComm) {
      if (!isSendRecvSelf) {
        // For point-to-point communication, lower rank of the two will get
        // unique id.
        if (rank_ == 0 || (singleP2POp && p2pRank == 0)) {
          C10D_MCCL_CHECK(mcclGetUniqueId(&mcclID), std::nullopt);
        }

        // Broadcast so that each process can have a unique MCCL ID
        auto timeStarted = std::chrono::steady_clock::now();
        broadcastUniqueMCCLID(&mcclID, singleP2POp, deviceKey, p2pRank);
        auto timerDeltaMs =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - timeStarted)
                .count() *
            1000;
        LOG(INFO) << logPrefix()
                  << "ProcessGroupMCCL broadcast unique ID through store took "
                  << timerDeltaMs << " ms";
      }

      // TODO: mccl has no config
      mcclComm = MCCLComm::create(numRanks, rank, mcclID, deviceIndex);
    }
  }

  // Creates the MCCL streams
  bool force_high = getCvarBool(TORCH_MCCL_HIGH_PRIORITY, false);
  auto streamVal = at::musa::getStreamFromPool(
      options_->is_high_priority_stream || force_high);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    inInitializationCommMap_.emplace(deviceKey, mcclComm);
  }

  FlightRecorder::get()->record_pg_ranks(
      std::make_tuple(pg_uid_, pg_desc_), groupRanks());

  VLOG(2) << logPrefix() << "ProcessGroupMCCL created mcclComm_ "
          << mcclComm->repr()
          << " on MUSA device: " << static_cast<int>(deviceIndex);

  // At this point MCCL should have been initialized, hence we can accurately
  // get the env value even if MCCL sets it by reading from mccl.conf file
  LOG(INFO) << logPrefix()
            << "MCCL_DEBUG: " << getCvarString({"MCCL_DEBUG"}, "N/A");

  // See [Group Start/End Note]
  for (const auto i : c10::irange(mcclActiveGroupCounter_)) {
    (void)i;
    C10D_MCCL_CHECK(mcclGroupStart(), std::nullopt);
  }

  mcclStreams_.emplace(deviceKey, streamVal);

  // Note: these events are created with the (default) cudaEventDisableTiming
  // flag This flag provides the best performance when used with
  // cudaStreamWaitEvent() and cudaEventQuery(). Since we here don't measure the
  // performance using cudaEvent, this should be set.
  // TODO(kwen2501): is ncclEvents_ used anywhere else?
  mcclEvents_.emplace(deviceKey, at::musa::MUSAEvent(musaEventDisableTiming));

  // Move the MCCL resource to cache
  auto it = inInitializationCommMap_.find(deviceKey);
  // A previous thread could've already removed devicesKey from
  // inInitializationCommMap_ and added it to devNCCLCommMap_
  if (it != inInitializationCommMap_.end()) {
    devMCCLCommMap_.emplace(deviceKey, std::move(it->second));
    inInitializationCommMap_.erase(deviceKey);

    // Now ncclComms are fully initialized.
    // Register all active MUSA memory segments in cache allocator to
    // the new MCCL communicators
    if (useTensorRegisterAllocatorHook_) {
      auto snapshot = c10::musa::MUSACachingAllocator::snapshot();
      // Register the segment to a new MCCL communicator if on the same device
      for (const auto& segmentInfo : snapshot.segments) {
        TORCH_INTERNAL_ASSERT(
            segmentInfo.device == device.index(),
            "Mismatch between MUSA memory segment device and current device");
        mcclComm->registerSegment(
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            reinterpret_cast<void*>(segmentInfo.address),
            segmentInfo.total_size);
      }
    }
    // Record the mapping between ncclComm and device index so that later
    // register hook can register a newly allocated segment to communicators
    // on the same device.
    // NOTE: we need remove the communicator from this map when it is
    // destroyed, otherwise may register onto an invalid communicator.
    mcclCommDevIdxMapMutex.lock();
    mcclCommDevIdxMap.emplace(mcclComm, device.index());
    mcclCommDevIdxMapMutex.unlock();
  }

  it = devMCCLCommMap_.find(deviceKey);
  TORCH_INTERNAL_ASSERT(
      it != devMCCLCommMap_.end(), "Communicators not populated in cache!");
  return it->second;
}

std::shared_ptr<MCCLComm> ProcessGroupMCCL::getMCCLComm(
    const std::string& deviceKey) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (devMCCLCommMap_.find(deviceKey) != devMCCLCommMap_.end()) {
    // Reuse the cached communicator if there is one.
    return devMCCLCommMap_[deviceKey];
  }
  return nullptr;
}

uint64_t ProcessGroupMCCL::getCommSplitCounter() const {
  uint64_t ret = 0;
  for (const auto& i : devMCCLCommMap_) {
    auto& mcclComm = i.second;
    ret += mcclComm->getCommSplitCounter();
  }
  return ret;
}

namespace {

// Check validity of tensor
void check_gpu_single_tensor(
    const at::Tensor& tensor,
    const bool p2p = false // whether operation is a P2P operation
) {
  if (!tensor.is_musa() || tensor.is_sparse()) {
    TORCH_CHECK(false, "Tensors must be MUSA and dense");
  }
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    if (p2p) {
      TORCH_WARN_ONCE(
          "Detected non-contiguous tensor in P2P operations. It is user "
          "responsibility to guarantee that source and destination tensors have "
          "the same contiguity format.");
    } else {
      TORCH_CHECK(false, "Tensors must be contiguous");
    }
  }
}

// Checks that all `tensors' have the same type and shape and reside on the same
// GPU.
// TODO: test_c10d_mccl.py should consider adding tests for the error conditions
// here, ie, that deliberately pass invalid tensors and check the right
// exception is thrown. The "Expected list of tensors on the same device"
// condition may be a challenge because the test would need to pass tensors on
// different devices in the same process.
int64_t check_gpu_tensors_same_device(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    TORCH_CHECK(false, "Tensor list must be nonempty");
  }

  const auto& first = tensors.front();

  int64_t total_numel = 0;
  for (const auto& t : tensors) {
    if (!t.is_musa() || t.is_sparse()) {
      TORCH_CHECK(false, "Tensors must be MUSA and dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      TORCH_CHECK(false, "Tensors must have identical type");
    }
    if (!t.is_non_overlapping_and_dense()) {
      TORCH_CHECK(false, "Tensors must be non-overlapping and dense");
    }
    // If we're in this function, the user called a _coalesced collective
    // on a set of tensors with potentially different sizes and strides.
    // Therefore, we don't check for matching sizes and strides,
    // but we do double-check tensors are on the same device.
    TORCH_CHECK(
        t.get_device() == tensors[0].get_device(),
        "Expected list of tensors on the same device");
    total_numel += t.numel();
  }

  return total_numel;
}

bool check_same_size(const std::vector<at::Tensor>& input_tensors) {
  for (const auto& input_tensor : input_tensors) {
    if (!input_tensors[0].is_same_size(input_tensor)) {
      return false;
    }
  }
  return true;
}

} // namespace

// RAII helper class to manage MCCL group API and MUSA free mutex.
// The destructor is allowed to throw since this helper class only
// manages group and lock lifetimes.
struct AutoMcclGroup {
  AutoMcclGroup();
  AutoMcclGroup(mcclComm_t comm, bool comm_nonblocking);
  ~AutoMcclGroup() noexcept(false);
  mcclComm_t comm_;
  bool comm_nonblocking_;
};

AutoMcclGroup::AutoMcclGroup() : comm_(nullptr), comm_nonblocking_(false) {
  C10D_MCCL_ASSERT(mcclGroupStart());
}

AutoMcclGroup::AutoMcclGroup(mcclComm_t comm, bool comm_nonblocking)
    : comm_(comm), comm_nonblocking_(comm_nonblocking) {
  C10D_MCCL_ASSERT(mcclGroupStart());
}

// NOLINTNEXTLINE(bugprone-exception-escape)
AutoMcclGroup::~AutoMcclGroup() noexcept(false) {
  // TODO: add non blocking mode
  C10D_MCCL_ASSERT(mcclGroupEnd());
}

c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL> ProcessGroupMCCL::initWork(
    at::Device& device,
    int rank,
    OpType opType,
    bool isP2P,
    const char* profilingTitle,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs, // TODO(kwen2501): necessary?
    bool record) {
  auto r = c10::make_intrusive<ProcessGroupMCCL::WorkMCCL>(
      pg_uid_,
      pg_desc_,
      device,
      rank,
      opType,
      isP2P ? seqP2P_ : seqCollective_,
      isP2P,
      profilingTitle,
      profilingTitle != nullptr ? std::optional<std::vector<at::Tensor>>(inputs)
                                : std::nullopt,
      desyncDebug_,
      enableTiming_.load(),
      musaEventCacheEnabled_.load(),
      dist_debug_level_);
  if (record) {
    bool isP2P = isP2POp(opType);
    // Ideally record every work that we enqueue, rather than every work we
    // create.
    // - at the time of this PR we do not currently enqueue every created work
    // - but it is unsafe to steal refs to start/end musa events from Works that
    //   may go out of scope before flight recorder has retired them,
    //   so we must ensure that any work that is initialized via initWork will
    //   be enqueued
    // - initially, moved record() into workEnqueue(), but found that makes it
    //   hard to get access to profilingTitle,
    //   inputs, and outputs for metadata recording, and we don't want to attach
    //   these objects to the Work becuase it has implications for keeping those
    //   tensors alive longer and adds overhead when copying Work objects
    //   between threads
    r->trace_id_ = FlightRecorder::get()->record(
        local_id_,
        std::make_tuple(pg_uid_, pg_desc_),
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle ? profilingTitle : "",
        inputs,
        outputs,
        r->mcclStartEvent_.get(),
        r->mcclEndEvent_.get(),
        options_->timeout,
        pgStatus_,
        isP2P);
  }
  return r;
}

std::vector<at::Tensor> ProcessGroupMCCL::WorkMCCL::result() {
  return *outputs_;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupMCCL::WorkMCCL::
    getFuture() {
  return future_;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupMCCL::WorkMCCL::
    getFutureResult() {
  return futureWorkResult_;
}

float ProcessGroupMCCL::WorkMCCL::getDuration() const {
  TORCH_CHECK(timingEnabled_, "getDuration only works if timing was enabled");
  TORCH_CHECK(
      mcclStartEvent_,
      "getDuration only works if mcclStartEvents_ is populated, true if timing enabled");
  TORCH_CHECK(
      mcclEndEvent_,
      "getDuration only works if mcclEndEvents_ is populated, which should always be true");
  return mcclStartEvent_->elapsed_time(*mcclEndEvent_);
}

uint64_t ProcessGroupMCCL::WorkMCCL::getSequencenumber() const {
  return seq_;
}

void ProcessGroupMCCL::assignTimeoutToWork(
    const c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work,
    const c10::intrusive_ptr<ProcessGroupMCCL::Options>& option) {
  std::chrono::milliseconds timeout = option->timeout;
  std::lock_guard<std::mutex> timeoutLock(mtxTimeoutExtension_);
  if (ephemeralTimeoutActive_.count() > 0) {
    timeout += ephemeralTimeoutActive_;
  }
  work->opTimeout_ = timeout;
  work->ownedEphermeralTimeout_ =
      ephemeralTimeoutActive_ - ephemeralTimeoutInflight_;
  ephemeralTimeoutInflight_ = ephemeralTimeoutActive_;
}

void ProcessGroupMCCL::workEnqueue(
    c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL> work) {
  // in blockingWait_ mode, we don't need watchdog thread, so no need to enqueue
  // the work
  if (!terminateProcessGroup_.load() && !blockingWait_) {
    std::lock_guard<std::mutex> lock(workMetaListMutex_);
    // Avoid view tensors to be processed in cleanup thread.
    // View tensors' destruction invokes autograd_meta, which
    // needs to be destructed in user thread. Otherwise will
    // get deadlock. Here we enqueue work without outputs_.
    workMetaList_.emplace_back(*work);
    // update the PG status related to the last enqueued work
    pgStatus_->lastEnqueuedSeq = static_cast<int64_t>(work->seq_);
    pgStatus_->lastEnqueuedWorkName = opTypeToString(work->opType_);
    pgStatus_->lastEnqueuedNumelIn = work->numelIn_;
    pgStatus_->lastEnqueuedNumelOut = work->numelOut_;
    lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
  }
}

ProcessGroupMCCL::Options::Options(bool is_high_priority_stream)
    : Backend::Options(MCCL_BACKEND_NAME, kProcessGroupMCCLDefaultTimeout),
      is_high_priority_stream(is_high_priority_stream) {}

static constexpr int CoalActive = 0x01, CoalColl = 0x02, CoalP2P = 0x04;

void ProcessGroupMCCL::startCoalescing() {
  // Other collective ops bump seq_ before creating a work. Thus, if coalesced
  // ops bump seq_ only after initing a work they will collide with (reuse) the
  // seq_ of the last non-coalesced collective.  Previously, seq_ was bumped
  // inside endCoalescing, but before initWork. Since we now record individual
  // ops from a coalesce group into the flight recorder, we want to have the
  // same seq_ for those ops and its 'endCoalescing' op. Hence we bump during
  // start, which has one minor downside- we burn a seq_ if someone ever does a
  // 'start' and 'end' coalescing region without doing an operation inbetween.

  coalescedDevice_.set_index(-1);
  coalescedComm_ = nullptr;
  coalescing_state_ |= CoalActive;
  groupStart();
}

// `optype` is for specifying a composite optype, such as ALLGATHER and
// REDUCE_SCATTER
c10::intrusive_ptr<Work> ProcessGroupMCCL::endCoalescing(OpType optype) {
  if (coalescedComm_ == nullptr) {
    // There is no actual work being coalesced, return here
    groupEnd();
    coalescing_state_ = 0;
    return nullptr;
  }
  TORCH_CHECK(
      coalescedDevice_.index() >= 0,
      "Somthing went wrong. Did you call end_coalescing before start_coalescing?");
  // `coalescedComm_` should have same set of comms across collectives
  auto comm = coalescedComm_;
  // `coalescedDevice_` should have same set of devices across collectives
  auto device = coalescedDevice_;

  // `getKeyFromDevice` is how we get keys for both collectives and batch P2P
  const auto key = getKeyFromDevice(device);
  auto mcclStream = mcclStreams_.at(key);

  // Create Work object
  c10::musa::CaptureStatus capture_status =
      c10::musa::currentStreamCaptureStatusMayInitCtx();
  bool enqueue =
      (coalescing_state_) && capture_status == c10::musa::CaptureStatus::None;
  auto work = initWork(
      device,
      rank_,
      optype,
      coalescing_state_ & CoalP2P,
      "mccl:coalesced",
      {},
      {},
      enqueue);
  work->mcclComm_ = comm;
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams_;
  work->store_ = store_;
  assignTimeoutToWork(work, options_);

  // Record start before ncclGroupEnd
  if (work->timingEnabled_) {
    work->mcclStartEvent_->record(mcclStream);
  }

  if (useNonblocking()) {
    groupEndNonblocking(comm);
  } else {
    groupEnd();
  }

  // Record end after ncclGroupEnd
  // TODO(eqy): is this still necessary if avoidRecordStreams_ is set?
  work->mcclEndEvent_->record(mcclStream);

  if (avoidRecordStreams_) {
    // other functions expect an initialized ptr if avoidRecordStreams_ is set
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>();
  }

  if (enqueue) {
    workEnqueue(work);
  }

  coalescing_state_ = 0;
  coalescedComm_ = nullptr;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::endCoalescing() {
  // Default OpType to COALESCED if not specified
  return endCoalescing(OpType::COALESCED);
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupMCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType,
    const char* profilingTitle,
    bool avoidRecordStreams,
    bool nanCheck) {
  // Environment setting by the user may add onto collective call's option
  avoidRecordStreams |= avoidRecordStreams_;
  nanCheck &= enableNanCheck_;

  auto device = getDevice(inputs[0]);
  // Guard must be created before `currentStreamCaptureStatusMayInitCtx`;
  // otherwise, extra MUSA context could be created on device 0.
  at::musa::OptionalMUSAGuard gpuGuard(device);

  c10::musa::CaptureStatus capture_status =
      c10::musa::currentStreamCaptureStatusMayInitCtx();
  errorIfCapturingNonCapturableMCCL();

  // Bump collective counter
  if (!coalescing_state_) {
    seqCollective_++;
  }
  op_id_++;

  const auto key = getKeyFromDevice(device);
  std::shared_ptr<MCCLComm> mcclComm = getMCCLComm(key);
  if (mcclComm == nullptr) {
    mcclComm = initMCCLComm(key, device, opType);
  }

  if (coalescing_state_ & CoalActive) {
    if ((coalescing_state_ & CoalColl) == 0) {
      // First op in coalesced operations
      seqCollective_++;
    }
    coalescing_state_ |= CoalColl;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = mcclComm;
    } else {
      TORCH_CHECK(coalescedComm_ == mcclComm, MULTI_DEVICE_ERROR_MSG);
    }
  }

  // Used many times below, so we stash the unordered_map lookup
  auto mcclStream = mcclStreams_.at(key);

  // First let MCCL streams wait for input tensors allocation streams
  syncStream(device, mcclEvents_[key], mcclStream);

  bool enqueue =
      !coalescing_state_ && capture_status == c10::musa::CaptureStatus::None;
  auto work = initWork(
      device, rank_, opType, false, profilingTitle, inputs, outputs, enqueue);

  // Store references to outputs to be used by WorkNCCL::result and operator<<.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  if (avoidRecordStreams) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>(inputs);
  }

  // TODO(chen.feng): enable nanCheck
  // if (nanCheck) {
  //   for (const auto& input : inputs) {
  //     checkForNan(input, mcclStream);
  //   }
  // }

  // Start event should only be recorded before the ncclGroupStart()
  if (work->timingEnabled_) {
    work->mcclStartEvent_->record(mcclStream);
  }

  pre(mcclStream, work);

  mcclComm_t comm = mcclComm->getMcclComm();

  // Both `inputs' and `outputs' are created on a worker stream and used in
  // different ncclStreams.  Hence, both must record the ncclStream to
  // prevent being freed before the collective finishes.
  //
  // We only record `inputs' here, and leave recording `outputs' to `fn' for
  // operations where `inputs' and `outputs' are not the same.
  //
  // See [Sync Streams].
  if (!avoidRecordStreams) {
    for (const auto& input : inputs) {
      if (!input.is_sparse()) {
        c10::musa::MUSACachingAllocator::recordStream(
            input.storage().data_ptr(), mcclStream);
      } else {
        // for sparse input case record streams on both index and value
        // tensors
        c10::musa::MUSACachingAllocator::recordStream(
            input.values().storage().data_ptr(), mcclStream);
        c10::musa::MUSACachingAllocator::recordStream(
            input.indices().storage().data_ptr(), mcclStream);
      }
    }
  }

  // Not all collectives have the same signature, e.g, all-reduce take in a
  // Tensor as the input and output while all-to-all take in a vector of Tensors
  // as input and output. Because we define the signature of the fn to take only
  // single tensor as input and output, we need to do a hack to get the first
  // element in the vector and pass it to fn.
  // TODO: we should clean up this in future (by either entirely removing
  // lambda's or removing input and output from lambda's signature).
  // TODO: add non blocking mode
  C10D_MCCL_CHECK(
      fn(inputs[0], outputs[0], comm, mcclStream),
      mcclComm->getMcclCommFailureReason());

  post(mcclStream, work);

  // End event should only be recorded after the mcclGroupEnd()
  if (!coalescing_state_) {
    work->mcclEndEvent_->record(mcclStream);
  }
  work->mcclComm_ = mcclComm;

  {
    c10::musa::MUSAMultiStreamGuard streamGuard(mcclStream);
    std::vector<at::Device> devices{device};
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks. wrapCallback() in MUSA
    // future blocks the stream this callback runs on the corresponding
    // ncclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false allows us to skip synchronization in
          // ivalue::Future, but is only valid as long as the lambda doesn't use
          // the "Future" argument.
          /*uses_future=*/false);
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Set appropriate work parameters.
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams;
  work->store_ = store_;
  assignTimeoutToWork(work, options_);
  // Record size info for debug. We only record the size on the first device as
  // multi-device per process is deprecated
  work->numelIn_ = 0;
  work->numelOut_ = 0;
  for (const auto& input : inputs) {
    work->numelIn_ += input.numel();
  }
  for (const auto& output : outputs) {
    work->numelOut_ += output.numel();
  }

  if (enqueue) {
    workEnqueue(work);
  }

  return work;
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupMCCL::collectiveCoalesced(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    OpType opType,
    const char* profilingTitle,
    bool avoidRecordStreams) {
  // Environment setting by the user may add onto collective call's option
  avoidRecordStreams |= avoidRecordStreams_;

  // Currently, the API permits one scenario where inputs.size() and
  // outputs.size() are > 0.
  // 1. If the call was a _coalesced call, all inputs must be on the same
  // device.
  //    The group of mccl calls applies the collective separately to each input,
  //    but the group as a whole should be efficient, and might even execute as
  //    a single fused kernel.
  auto device = getDevice(inputs[0]);
  // Guard must be created before `currentStreamCaptureStatusMayInitCtx`;
  // otherwise, extra MUSA context could be created on device 0.
  at::musa::OptionalMUSAGuard gpuGuard(device);

  c10::musa::CaptureStatus capture_status =
      c10::musa::currentStreamCaptureStatusMayInitCtx();
  errorIfCapturingNonCapturableMCCL();

  // Bump collective counter
  seqCollective_++;

  // For coalescingManager collectives, there is no individual c++ call per
  // collective so there is no flight record and we increment seqCollective_ and
  // op_id_ together. Compare this to startCoalescing/endCoalescing flow where
  // we increment either seqP2P_ or seqCollective_ once per group and increment
  // op_id_ once per indvidual operation within the group
  op_id_++;

  const auto key = getKeyFromDevice(device);
  std::shared_ptr<MCCLComm> mcclComm = getMCCLComm(key);
  if (mcclComm == nullptr) {
    mcclComm = initMCCLComm(key, device, opType);
  }

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalColl;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = mcclComm;
    } else {
      TORCH_CHECK(coalescedComm_ == mcclComm, MULTI_DEVICE_ERROR_MSG);
    }
  }

  // Used many times below, so we stash the unordered_map lookup
  auto mcclStream = mcclStreams_.at(key);

  // First let MCCL streams wait for input tensors allocation streams
  syncStream(device, mcclEvents_[key], mcclStream);

  auto work = initWork(
      device,
      rank_,
      opType,
      false,
      profilingTitle,
      inputs,
      outputs,
      /*record=*/true);

  // Store references to outputs to be used by WorkNCCL::result and operator<<.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  if (avoidRecordStreams) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>(inputs);
  }

  // Start event should only be recorded before the ncclGroupStart() (which
  // happens inside AutoNcclGroup guard below)
  if (work->timingEnabled_) {
    work->mcclStartEvent_->record(mcclStream);
  }

  mcclComm_t comm = mcclComm->getMcclComm();

  // if (enableCollecticeHashDebug_.load()) {
  //   auto numel = getTensorsNumel(inputs);
  //   auto hashValue = hashTensors(inputs);
  //   PRINT_COLLECTIVE_HASH_SIGNATURE(
  //       "input", opTypeToString(opType), numel, hashValue);
  // }

  {
    AutoMcclGroup mccl_group_guard(comm, useNonblocking());
    for (const auto i : c10::irange(inputs.size())) {
      // Both `inputs' and `outputs' are created on a worker stream and used in
      // different ncclStreams.  Hence, both must record the ncclStream to
      // prevent being freed before the collective finishes.
      //
      // We only record `inputs' here, and leave recording `outputs' to `fn' for
      // operations where `inputs' and `outputs' are not the same.
      //
      // See [Sync Streams].
      if (!avoidRecordStreams) {
        if (!inputs[i].is_sparse()) {
          c10::musa::MUSACachingAllocator::recordStream(
              inputs[i].storage().data_ptr(), mcclStream);
        } else {
          // for sparse input case record streams on both index and value
          // tensors
          c10::musa::MUSACachingAllocator::recordStream(
              inputs[i].values().storage().data_ptr(), mcclStream);
          c10::musa::MUSACachingAllocator::recordStream(
              inputs[i].indices().storage().data_ptr(), mcclStream);
        }
      }
      // TODO: add non blocking mode
      C10D_MCCL_CHECK(
          fn(inputs[i], outputs[i], comm, mcclStream),
          mcclComm->getMcclCommFailureReason());
    }
  }

  work->mcclEndEvent_->record(mcclStream);
  work->mcclComm_ = mcclComm;

  {
    c10::musa::MUSAMultiStreamGuard streamGuard(mcclStream);
    std::vector<at::Device> devices{device};
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks. wrapCallback() in MUSA
    // future blocks the stream this callback runs on the corresponding
    // ncclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false allows us to skip synchronization in
          // ivalue::Future, but is only valid as long as the lambda doesn't use
          // the "Future" argument.
          /*uses_future=*/false);
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Set appropriate work parameters.
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams;
  work->store_ = store_;
  assignTimeoutToWork(work, options_);
  // Record size info for debug. We only record the size on the first device as
  // multi-device per process is deprecated
  work->numelIn_ = inputs[0].numel();
  work->numelOut_ = outputs[0].numel();

  /* Note [musa graph capture and workEnqueue]

  Normal behavior of the C10D watchdog is to query musa events on work objects.
  We disable this event query behavior during graph capture as it is disallowed
  during capture under the strictest capture mode setting.
  Note that previously recorded events (e.g., before the capture) can be queried
  as the watchdog capture mode has been changed to thread-local, but user-side
  event queries (from the main thread) via .is_completed() are still disallowed.
  TODO(eqy): Is there a path to allowing workEnqueue during graph capture for
  watchdog-thread usage only?

  TODO:
   - Is our design for flight recorder safe in this context?  are we recording
  any FR events during cudagraph capture? if so, they won't be safe to poll for
  completion status.
  */
  if (capture_status == c10::musa::CaptureStatus::None) {
    workEnqueue(work);
  }
  // TODO(whc) if the work isn't enqueued, I don't feel great about returning
  // it, since interactions with it by usercode won't behave normally - they
  // won't observe work completion, for instance.  Will this lead to silent
  // problems during capture?
  return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupMCCL::pointToPoint(
    at::Tensor& tensor,
    Fn fn,
    int peer,
    OpType opType,
    PreProcess pre,
    PostProcess post,
    const char* profilingTitle) {
  // avoidRecordStreams_ note:
  // send, recv, and irecv should be ok with avoidRecordStreams,
  // However, for isend, I don't think the API requires the user
  // to wait() on the returned handle, so ProcessGroupMCCL can't know
  // when it's safe to release the input back to the allocator,
  // and the present call has no way to know it's not an isend.
  // Therefore, we warn and fall back to the typical recordStream logic:
  if (avoidRecordStreams_) {
    TORCH_WARN_ONCE(
        "TORCH_MCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point ",
        "collectives.");
  }

  auto device = getDevice(tensor);
  at::musa::OptionalMUSAGuard gpuGuard(device);

  std::string key;
  int p2pRank = 0, p2pTargetRank = 0;
  bool isSendRecvSelf = false;
  // For batch_isend_irecv, mcclGroupStart() would be called upfront
  bool batchP2P = mcclActiveGroupCounter_ > 0;
  if (batchP2P) {
    // For batch P2P, we need to treat it like a collective when selecting
    // communicator, because other ranks can call into this batch other than my
    // rank and my peer
    key = getKeyFromDevice(device);
    p2pRank = rank_;
    p2pTargetRank = peer;
  } else {
    // For single P2P, preserve the old two-rank behavior (to avoid perf diff)
    key = getKeySendRecv(rank_, peer);
    p2pRank = rank_ <= peer ? 0 : 1;
    isSendRecvSelf = rank_ == peer;
    p2pTargetRank = isSendRecvSelf ? 0 : 1 - p2pRank;

    if (!coalescing_state_) {
      // Bump P2P sequence number.
      seqP2P_++;
    }
  }

  // Bump the logical operation counter regardless of whether this op is
  // coalesced or individual
  op_id_++;

  std::shared_ptr<MCCLComm> mcclComm = getMCCLComm(key);
  if (mcclComm == nullptr) {
    mcclComm = initMCCLComm(key, device, opType, p2pRank, isSendRecvSelf);
  }

  if (coalescing_state_ & CoalActive) {
    // Bump  seqP2P_ once per coalesced group, not once per individual op.
    if ((coalescing_state_ & CoalP2P) == 0) {
      seqP2P_++;
    }
    coalescing_state_ |= CoalP2P;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = mcclComm;
    } else {
      TORCH_CHECK(coalescedComm_ == mcclComm, MULTI_DEVICE_ERROR_MSG);
    }
  }

  // Used many times below, so we stash the unordered_map lookup
  auto mcclStream = mcclStreams_.at(key);
  // First let MCCL streams wait for input tensors allocation streams
  syncStream(device, mcclEvents_[key], mcclStream);

  // Work itself will create the MUSA events on all GPUs of tensors
  c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL> work;
  if (coalescing_state_) {
    // When coalescing, we record events per op that lack timing/state
    // information becuase there is no 'work' associated with them, and then
    // later in endCoalescing we record a 'coalesced' Work which has
    // timing/state updates via watchdog thread, but lacks op metadata such as
    // input/output sizes and profilingTitle per-op in the group.
    auto trace_id = FlightRecorder::get()->record(
        local_id_,
        std::make_tuple(pg_uid_, pg_desc_),
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle,
        {tensor},
        {tensor},
        nullptr,
        nullptr,
        options_->timeout,
        pgStatus_,
        /*isP2P=*/true);
    // TODO(whc) if we want to make the per-p2p-op flightrecorder entries get
    // their timings/states updated by proxy when the Work obj representing the
    // coalesce group gets its update, we could accumulate these trace_ids
    // together and ask FlightRecorder to take the update from one Work and
    // apply it to multiple entries
    (void)trace_id;
  } else {
    // Store references to outputs to be used by WorkNCCL::result and
    // operator<<. Note that these outputs are only valid for recv(), as send()
    // does not modify the inputs but we still create these outputs for use
    // cases such as profiling.

    work = initWork(
        device,
        rank_,
        opType,
        true,
        profilingTitle,
        {tensor},
        {},
        /*record=*/false);
    // This bypasses something in Work() that crashes if {tensor} is given as
    // output, not sure what
    work->outputs_ = std::make_shared<std::vector<at::Tensor>>();
    work->outputs_->push_back(tensor);
    // TODO(whc) because we don't pass output {tensor} to initWork, we tell
    // initWork to not record, and then we manually call record passing all the
    // information it wants.
    work->trace_id_ = FlightRecorder::get()->record(
        local_id_,
        std::make_tuple(pg_uid_, pg_desc_),
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle,
        {tensor},
        {tensor},
        work->mcclStartEvent_.get(),
        work->mcclEndEvent_.get(),
        options_->timeout,
        pgStatus_,
        /*isP2P=*/true);
  }

  // Only check for NaN for send ops, for recv ops `tensor` can be a random
  // placeholder
  // TODO(chen.feng): enable nanCheck
  // if (enableNanCheck_ && opType == OpType::SEND) {
  //   checkForNan(tensor, mcclStream);
  // }

  if (!coalescing_state_) {
    // Start event should only be recorded before the ncclGroupStart()
    if (work->timingEnabled_) {
      work->mcclStartEvent_->record(mcclStream);
    }

    pre(mcclStream, work);
  }

  // Both send tensor and recv tensor are created on a worker stream and used
  // in different ncclStreams.  Hence, both must record the ncclStream to
  // prevent being freed before the collective finishes.
  //
  // See [Sync Streams].
  c10::musa::MUSACachingAllocator::recordStream(
      tensor.storage().data_ptr(), mcclStream);

  // This part seems common to both p2p and coalesced-p2p usage?
  mcclComm_t comm_ = mcclComm->getMcclComm();

  // TODO: add nonblocking mode
  C10D_MCCL_CHECK(
      fn(tensor, comm_, mcclStream, p2pTargetRank),
      mcclComm->getMcclCommFailureReason());

  if (!coalescing_state_) {
    post(mcclStream);

    // End event should only be recorded after the ncclGroupEnd()
    work->mcclEndEvent_->record(mcclStream);
    work->mcclComm_ = mcclComm;
    work->blockingWait_ = blockingWait_;
    work->store_ = store_;
    assignTimeoutToWork(work, options_);
    // Record size info for debug. We only record the size on the first device
    // as multi-device per process is deprecated
    work->numelIn_ = work->numelOut_ = tensor.numel();

    // Future only needs to be created and marked completed with outputs for
    // recv(), but still create future for use cases such as profiling even for
    // send().
    {
      c10::musa::MUSAMultiStreamGuard streamGuard(mcclStream);
      std::vector<at::Device> devices{device};
      work->future_ = c10::make_intrusive<at::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()), devices);
      work->future_->markCompleted(at::IValue(*work->outputs_));
    }

    // Add a callback that runs profiling end callbacks. wrapCallback() in MUSA
    // future blocks the stream this callback runs on the corresponding
    // ncclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false allows us to skip synchronization in
          // ivalue::Future, but is only valid as long as the lambda doesn't use
          // the "Future" argument.
          /*uses_future=*/false);
    }
  }

  // Enqueue P2P op so that it can be cancelled by MCCL watchdog
  c10::musa::CaptureStatus capture_status =
      c10::musa::currentStreamCaptureStatusMayInitCtx();

  if (!coalescing_state_ && capture_status == c10::musa::CaptureStatus::None) {
    workEnqueue(work);
  }
  return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupMCCL::collective(
    at::Tensor& input,
    at::Tensor& output,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType,
    const char* profilingTitle,
    bool avoidRecordStreams,
    bool nanCheck) {
  auto inputs = std::vector<at::Tensor>{input};
  auto outputs = std::vector<at::Tensor>{output};
  return collective(
      inputs,
      outputs,
      fn,
      pre,
      post,
      opType,
      profilingTitle,
      avoidRecordStreams,
      nanCheck);
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupMCCL::collective(
    at::Tensor& input,
    at::Tensor& output,
    Fn fn,
    OpType opType,
    const char* profilingTitle,
    bool avoidRecordStreams,
    bool nanCheck) {
  auto inputs = std::vector<at::Tensor>{input};
  auto outputs = std::vector<at::Tensor>{output};
  return collective(
      inputs,
      outputs,
      fn,
      [](at::musa::MUSAStream&,
         c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {},
      [](at::musa::MUSAStream&,
         c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {},
      opType,
      profilingTitle,
      avoidRecordStreams,
      nanCheck);
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupMCCL::pointToPoint(
    at::Tensor& tensor,
    Fn fn,
    int peer,
    OpType opType,
    const char* profilingTitle) {
  return pointToPoint(
      tensor,
      fn,
      peer,
      opType,
      [](at::musa::MUSAStream&,
         c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {},
      [](at::musa::MUSAStream&) {},
      profilingTitle);
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::allreduce_impl(
    at::Tensor& tensor,
    const char* profilingTitle,
    const AllreduceOptions& opts) {
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream) {
        auto mcclDataType = getMcclDataType(input.scalar_type());
        auto mcclReduceOp =
            getMcclReduceOp(opts.reduceOp, input, mcclDataType, comm);
        return mcclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            mcclDataType,
            mcclReduceOp,
            comm,
            stream.stream());
      },
      OpType::ALLREDUCE,
      profilingTitle);
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    TORCH_CHECK(
        complexViewAsRealAllowed(opts.reduceOp),
        "all_reduce does not support",
        opts.reduceOp,
        "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  check_gpu_single_tensor(tensor);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return allreduce_impl(tensor, "mccl:all_reduce", opts);
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  auto total_numel = check_gpu_tensors_same_device(tensors);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective and assume only one collective
                  // in coalesed range
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce_coalesced", // collective name
      total_numel, // inNelems
      total_numel, // outNelems
      tensors[0].scalar_type(), // dType
      // I'm not sure what in,outSplitSizes mean here.
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return collectiveCoalesced(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          at::musa::MUSAStream& stream) {
        auto mcclDataType = getMcclDataType(input.scalar_type());
        auto mcclReduceOp =
            getMcclReduceOp(opts.reduceOp, input, mcclDataType, comm);
        return mcclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            mcclDataType,
            mcclReduceOp,
            comm,
            stream.stream());
      },
      OpType::COALESCED,
      "mccl:allreduce_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    tensor = at::view_as_real(tensor);
  }
  check_gpu_single_tensor(tensor);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      opts.rootRank, // root rank
      "broadcast", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  const auto root = opts.rootRank + opts.rootTensor;
  bool nanCheck = (root == rank_);

  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          at::musa::MUSAStream& stream) {
        return mcclBcast(
            input.data_ptr(),
            input.numel(),
            getMcclDataType(input.scalar_type()),
            static_cast<int>(root),
            comm,
            stream.stream());
      },
      OpType::BROADCAST,
      "mccl:broadcast",
      avoidRecordStreams,
      nanCheck);
}

// _broadcast_oop adds an out-of-place broadcast in PGMCCL
// Custom collectives may be implemented by coalescing broadcast operations
// One use-case is implementing a vector all_gather (all_gather_v)
// where unevenly sized inputs are gathered among participating ranks
// Since all_gather provides an out-of-place API, an all_gather_v
// semantic implemented inside pg_mccl.all_gather also needs to support
// out-of-place, for which an out-of-place broadcast is required to be added
c10::intrusive_ptr<Work> ProcessGroupMCCL::_broadcast_oop(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const BroadcastOptions& opts) {
  if (outputTensor.numel() != inputTensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "Tensor input and output of _broadcast_oop must have the same number of elements ");
  }
  const auto root = opts.rootRank + opts.rootTensor;
  bool nanCheck = (root == rank_);
  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          at::musa::MUSAStream& stream) {
        return mcclBroadcast(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getMcclDataType(input.scalar_type()),
            static_cast<int>(root),
            comm,
            stream.stream());
      },
      OpType::BROADCAST,
      "mccl:_broadcast_oop",
      /*avoidRecordStreams=*/false,
      nanCheck);
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    TORCH_CHECK(
        complexViewAsRealAllowed(opts.reduceOp),
        "reduce does not support",
        opts.reduceOp,
        "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  check_gpu_single_tensor(tensor);
  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      opts.rootRank, // root rank
      "reduce", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          at::musa::MUSAStream& stream) {
        const auto root = opts.rootRank + opts.rootTensor;
        auto mcclDataType = getMcclDataType(input.scalar_type());
        auto mcclReduceOp =
            getMcclReduceOp(opts.reduceOp, input, mcclDataType, comm);
        return mcclReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            mcclDataType,
            mcclReduceOp,
            static_cast<int>(root),
            comm,
            stream.stream());
      },
      OpType::REDUCE,
      "mccl:reduce");
}

// _reduce_oop exposes an out-of-place reduce from PGMCCL
// Custom collectives may be implemented by coalescing reduce operations
// One use-case is implementing a vector reduce_scatter (reduce_scatter_v)
// where inputs are reduced and scattered unevenly among participating ranks
// Since reduce_scatter provides an out-of-place API, a reduce_scatter_v
// semantic implemented inside pg_mccl.reduce_scatter also needs to support
// out-of-place, for which an out-of-place reduce is required to be added
c10::intrusive_ptr<Work> ProcessGroupMCCL::_reduce_oop(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceOptions& opts) {
  if (outputTensor.numel() != inputTensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "Tensor input and output of _reduce_oop must have the same number of elements ");
  }
  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          at::musa::MUSAStream& stream) {
        const auto root = opts.rootRank + opts.rootTensor;
        const auto mcclDataType = getMcclDataType(input.scalar_type());
        const auto mcclReduceOp =
            getMcclReduceOp(opts.reduceOp, input, mcclDataType, comm);
        return mcclReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            mcclDataType,
            mcclReduceOp,
            (int)root,
            comm,
            stream.stream());
      },
      OpType::REDUCE,
      "mccl:_reduce_oop");
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto inputTensor = inputTensors.back();
  check_gpu_single_tensor(inputTensor);
  auto outputTensors_ = outputTensors.back();

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "all_gather", // collective name
      inputTensor.numel(), // inNelems
      inputTensor.numel() * // outNelems
          this->getSize(),
      inputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  bool same_size = check_same_size(outputTensors_);
  if (same_size) {
    // Flatten a vector of tensors into a single, stacked tensor.
    at::Tensor outputFlattened = newLikeFlat(outputTensors_);

    return collective(
        inputTensor,
        outputFlattened,
        [&](at::Tensor& input,
            at::Tensor& output,
            mcclComm_t comm,
            at::musa::MUSAStream& stream) {
          if (!avoidRecordStreams_) {
            c10::musa::MUSACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          return mcclAllGather(
              input.data_ptr(),
              output.data_ptr(),
              input.numel(),
              getMcclDataType(input.scalar_type()),
              comm,
              stream.stream());
        },
        [](at::musa::MUSAStream& mcclStream,
           c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {
          // avoidRecordStreams_ note: We actually don't need to stash anything
          // here.
          //  - inputTensors is stashed onto work->stashed_for_allocator_safety_
          //    in collective().
          //  - outputFlattened is stashed onto work->outputs_ in collective().
          //  - User-facing outputTensors should be held by the user until after
          //    waiting on work_, or the call makes no sense.
          // So all participating tensors are accounted for, and won't be
          // released back to their allocation streams until after work_ is
          // waited on.
        },
        [&](at::musa::MUSAStream& mcclStream,
            c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {
          // Copy the flattened output tensors to the outputs.
          at::musa::MUSAStreamGuard guard(mcclStream);
          for (const auto j : c10::irange(outputTensors_.size())) {
            // See [Sync Streams].
            if (!avoidRecordStreams_) {
              c10::musa::MUSACachingAllocator::recordStream(
                  outputTensors_[j].storage().data_ptr(), mcclStream);
            }
            outputTensors_[j].copy_(
                outputFlattened[static_cast<int64_t>(j)], true);
          }
        },
        OpType::ALLGATHER,
        "mccl:all_gather");
  } else {
    const auto num_reduces = outputTensors_.size();
    startCoalescing();
    for (const int64_t i : c10::irange(static_cast<int64_t>(num_reduces))) {
      auto& output = outputTensors_[i];
      auto& input = (i == rank_) ? inputTensor : output;
      auto broadcastOpts = BroadcastOptions{i, int64_t(0), opts.timeout};
      _broadcast_oop(output, input, broadcastOpts);
    }
    auto work = endCoalescing(OpType::ALLGATHER);
    return work;
  }
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  TORCH_CHECK(false, "ProcessGroupMCCL does not support allgather_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective and assume only one collective
                  // in coalesed range
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputs, // inputTensors
      outputs, // outputTensors
      rank_, // rank
      "allgather_into_tensor_coalesced", // collective name
      getTensorsNumel(inputs), // inNelems
      getTensorsNumel(outputs), // outNelems
      inputs[0].scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  return collectiveCoalesced(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          at::musa::MUSAStream& stream) {
        return mcclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getMcclDataType(input.scalar_type()),
            comm,
            stream.stream());
      },
      OpType::COALESCED,
      "mccl:all_gather_into_tensor_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto outputTensor = outputTensors.back();
  check_gpu_single_tensor(outputTensor);
  auto inputTensors_ = inputTensors.back();
  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "reduce_scatter", // collective name
      outputTensor.numel() * this->getSize(), // inNelems
      outputTensor.numel(), // outNelems
      outputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  bool same_size = check_same_size(inputTensors_);
  if (same_size) {
    // Flatten a vector of tensors into a single, stacked tensor.
    at::Tensor inputFlattened = newLikeFlat(inputTensors_);

    return collective(
        inputFlattened,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            mcclComm_t comm,
            at::musa::MUSAStream& stream) {
          if (!avoidRecordStreams_) {
            c10::musa::MUSACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          const auto mcclDataType = getMcclDataType(input.scalar_type());
          const auto mcclReduceOp =
              getMcclReduceOp(opts.reduceOp, input, mcclDataType, comm);
          return mcclReduceScatter(
              input.data_ptr(),
              output.data_ptr(),
              output.numel(),
              mcclDataType,
              mcclReduceOp,
              comm,
              stream.stream());
        },
        [&](at::musa::MUSAStream& mcclStream,
            c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {
          if (avoidRecordStreams_) {
            // We only need to stash inputTensors.
            //  - inputFlattened is stashed onto
            //  work->stashed_for_allocator_safety_
            //    in collective().
            //  - User-facing outputTensors is stashed onto work->outputs_ in
            //  collective(),
            //    and should also be held by the user until after waiting on
            //    work_.
            auto& v = work->stashed_for_allocator_safety_;
            v->insert(v->end(), inputTensors_.begin(), inputTensors_.end());
          }

          // Copy the input tensors to the flattened inputs.
          at::musa::MUSAStreamGuard guard(mcclStream);
          for (const auto j : c10::irange(inputTensors_.size())) {
            // See [Sync Streams].
            if (!avoidRecordStreams_) {
              c10::musa::MUSACachingAllocator::recordStream(
                  inputTensors_[j].storage().data_ptr(), mcclStream);
            }
            inputFlattened[static_cast<int64_t>(j)].copy_(
                inputTensors_[j], true);
          }
        },
        [&](at::musa::MUSAStream&,
            c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {},
        OpType::REDUCE_SCATTER,
        "mccl:reduce_scatter");
  } else {
    const auto num_reduces = inputTensors_.size();
    startCoalescing();
    for (const int i : c10::irange(static_cast<int>(num_reduces))) {
      auto& input = inputTensors_[i];
      auto& output = (i == rank_) ? outputTensor : input;
      auto reduceOpts = ReduceOptions{
          opts.reduceOp,
          static_cast<int64_t>(i),
          static_cast<int64_t>(0),
          opts.timeout};
      _reduce_oop(output, input, reduceOpts);
    }
    auto work = endCoalescing(OpType::REDUCE_SCATTER);
    return work;
  }
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts) {
  if (inputTensor.dtype() != outputTensor.dtype()) {
    TORCH_CHECK(
        false, "input tensor must be the same type as the output tensor.");
  }

  if (inputTensor.numel() != outputTensor.numel() * size_) {
    TORCH_CHECK(
        false,
        "input tensor must be the same size as output size times world size");
  }

  // @lint-ignore CLANGTIDY
  const auto& tensor = outputTensor;

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensor, // inputTensor
      outputTensor, // outputTensor
      rank_, // rank
      "_reduce_scatter_base", // collective name
      inputTensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dtype
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // for asyncOp = false, we don't want to record streams because we
  // know that the MCCL stream will join back to the "current" stream right
  // after this op. So we might just as well keep the stream ownership of the
  // input/output tensors unchanged. The benefit would be that the
  // allocation/free of the tensors would look deterministic to the "current"
  // stream so that the caching allocator can reuse memory pool for this stream
  // in a clever way. This setting is added for libraries like FSDP which uses
  // `reduce_scatter_tensor`.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          at::musa::MUSAStream& stream) {
        if (!avoidRecordStreams) {
          c10::musa::MUSACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        auto mcclDataType = getMcclDataType(input.scalar_type());
        auto mcclReduceOp =
            getMcclReduceOp(opts.reduceOp, input, mcclDataType, comm);
        return mcclReduceScatter(
            input.data_ptr(),
            output.data_ptr(),
            output.numel(),
            mcclDataType,
            mcclReduceOp,
            comm,
            stream.stream());
      },
      OpType::_REDUCE_SCATTER_BASE,
      "mccl:_reduce_scatter_base",
      avoidRecordStreams);
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {
  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective and assume only one collective
                  // in coalesed range
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputs, // inputTensors
      outputs, // outputTensors
      rank_, // rank
      "reduce_scatter_tensor_coalesced", // collective name
      getTensorsNumel(inputs), // inNelems
      getTensorsNumel(outputs), // outNelems
      inputs[0].scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  return collectiveCoalesced(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          at::musa::MUSAStream& stream) {
        if (!avoidRecordStreams_) {
          c10::musa::MUSACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        auto mcclDataType = getMcclDataType(input.scalar_type());
        auto mcclReduceOp =
            getMcclReduceOp(opts.reduceOp, input, mcclDataType, comm);
        return mcclReduceScatter(
            input.data_ptr(),
            output.data_ptr(),
            output.numel(),
            mcclDataType,
            mcclReduceOp,
            comm,
            stream.stream());
      },
      OpType::COALESCED,
      "mccl:reduce_scatter_tensor_coalesced");
}

c10::DeviceIndex ProcessGroupMCCL::guessDeviceId() const {
  // 1st choice: don't use this function if your API can take a device_id
  // argument.
  if (getBoundDeviceId().has_value()) {
    // 2nd choice: Use the bound GPU device id if available.
    // Bounded device id can be passed to `init_process_group`.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return getBoundDeviceId().value().index();
  } else if (!usedDeviceIdxs_.empty()) {
    // 3rd choice: infer the device id from the used device ids.
    return *usedDeviceIdxs_.begin();
  }
  // This means there is not yet a MCCL collective being called
  // Here we have to use the best guesses and will use a single GPU to call
  // allreduce to achieve barrier.
  // In case the multiple processes fall into the same node, we use rank to
  // ensure that each process is on a different GPU
  // Note: it is better to use global rank because the group-local rank can be
  // offset wrt the device id if intra-node GPUs are sharded into multiple
  // dimensions.
  int devIdx = globalRank() % localDeviceCount_;
  LOG(WARNING)
      << logPrefix()
      << c10::str(
             " using GPU ",
             devIdx,
             " as device used by this process is currently unknown. ",
             "This can potentially cause a hang if this rank to GPU mapping is incorrect. ",
             "You can pecify device_id in init_process_group() to force use of a particular device.");
  return static_cast<c10::DeviceIndex>(devIdx);
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::barrier(const BarrierOptions& opts) {
  RECORD_PARAM_COMMS(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      rank_, // rank
      "barrier", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // Device to use for barrier
  c10::DeviceIndex barDevIdx = -1;

  // Select device to use for barrier
  // 1st choice: Use user defined GPU device ids if provided
  if (!opts.device_ids.empty()) {
    // Use the first device id because PG MCCL is single-device now
    barDevIdx = static_cast<c10::DeviceIndex>(opts.device_ids[0]);
  } else {
    // 2nd choice: Use the bound or used GPU device id if available.
    barDevIdx = guessDeviceId();
  }

  TORCH_CHECK_WITH(
      ValueError,
      barDevIdx >= 0,
      "Failed to infer a GPU device id to perform barrier. ");
  auto barDevice = at::Device(at::DeviceType::PrivateUse1, barDevIdx);

  // Create a dummy tensor on the device
  // Note: we use zeros() instead of empty() to prevent barrier from triggering
  // alarm when NaN checker is enabled.
  at::Tensor barrierTensor =
      at::zeros({1}, at::TensorOptions().device(barDevice).dtype(at::kFloat));

  // All reduce to achieve the barrier
  auto work = allreduce_impl(barrierTensor, "mccl:all_reduce_barrier");

  // Work will take over barrierTensors
  auto mcclWork = dynamic_cast<ProcessGroupMCCL::WorkMCCL*>(work.get());
  TORCH_CHECK(mcclWork);
  mcclWork->isBarrierOp_ = true;
  return work;
}

static mcclResult_t all2all_single_equal_split_impl(
    at::Tensor& input,
    at::Tensor& output,
    int size,
    mcclComm_t comm,
    c10::musa::MUSAStream& stream) {
  int numranks;
  auto type = getMcclDataType(input.scalar_type());
  size_t count = input.numel() / size;
  size_t rankdiff = input.nbytes() / size;
  const auto* sendbuff = reinterpret_cast<char*>(input.data_ptr());
  auto* recvbuff = reinterpret_cast<char*>(output.data_ptr());
  // TODO(yueran.tang): mccl has no ROCM or AllToAll operators.
  // Support it in the future.
  C10D_MCCL_ASSERT(mcclCommCount(comm, &numranks));
  C10D_MCCL_ASSERT(mcclGroupStart());
  for (const auto r : c10::irange(numranks)) {
    // MCCL uses 0 byte message for synchronization
    // Avoid send/recv when message size is zero
    if (count != 0) {
      C10D_MCCL_ASSERT(
          mcclSend(sendbuff + r * rankdiff, count, type, r, comm, stream));
      C10D_MCCL_ASSERT(
          mcclRecv(recvbuff + r * rankdiff, count, type, r, comm, stream));
    }
  }
  C10D_MCCL_ASSERT(mcclGroupEnd());
  return mcclSuccess;
}

static mcclResult_t all2all_single_unequal_split_impl(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    c10::ScalarType _type,
    mcclComm_t comm,
    c10::musa::MUSAStream& stream) {
  auto type = getMcclDataType(_type);
  int numranks;
  C10D_MCCL_ASSERT(mcclCommCount(comm, &numranks));
  C10D_MCCL_ASSERT(mcclGroupStart());
  for (const auto r : c10::irange(numranks)) {
    // MCCL uses 0 byte message for synchronization
    // Avoid send/recv when message size is zero
    if (sendcounts[r] != 0) {
      C10D_MCCL_ASSERT(mcclSend(
          ((char*)sendbuff) + senddispls[r] * size,
          sendcounts[r],
          type,
          r,
          comm,
          stream));
    }
    if (recvcounts[r] != 0) {
      C10D_MCCL_ASSERT(mcclRecv(
          ((char*)recvbuff) + recvdispls[r] * size,
          recvcounts[r],
          type,
          r,
          comm,
          stream));
    }
  }
  C10D_MCCL_ASSERT(mcclGroupEnd());
  return mcclSuccess;
}

static mcclResult_t all2all_impl(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    mcclComm_t comm,
    c10::musa::MUSAStream& stream) {
  C10D_MCCL_ASSERT(mcclGroupStart());
  for (const auto r : c10::irange(outputTensors.size())) {
    at::Tensor& input = inputTensors[r];
    at::Tensor& output = outputTensors[r];
    if (input.numel() != 0) {
      C10D_MCCL_ASSERT(mcclSend(
          input.data_ptr(),
          input.numel(),
          getMcclDataType(input.scalar_type()),
          r,
          comm,
          stream.stream()));
    }
    if (output.numel() != 0) {
      C10D_MCCL_ASSERT(mcclRecv(
          output.data_ptr(),
          output.numel(),
          getMcclDataType(output.scalar_type()),
          r,
          comm,
          stream.stream()));
    }
  }
  C10D_MCCL_ASSERT(mcclGroupEnd());
  return mcclSuccess;
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  check_gpu_single_tensor(outputTensor);
  check_gpu_single_tensor(inputTensor);
  if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
    RECORD_PARAM_COMMS_DATA(
        std::make_tuple(
            static_cast<int64_t>(seqCollective_) + 1,
            false), // seq + 1 to match collective
        std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_all", // collective name
        inputTensor.numel(), // inNelems
        outputTensor.numel(), // outNelems
        inputTensor.scalar_type(), // dType
        std::vector<int64_t>(), // inSplitSizes
        std::vector<int64_t>(), // outSplitSizes
        globalRankStart, // globalRankStart
        globalRankStride, // globalRankStride
        this->getSize()); // worldSize

    // avoidRecordStreams_ note: collective() will stash inputTensors and
    // outputTensors.
    return collective(
        inputTensor,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            mcclComm_t comm,
            c10::musa::MUSAStream& stream) {
          // See [Sync Streams].
          if (!avoidRecordStreams_) {
            c10::musa::MUSACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          return all2all_single_equal_split_impl(
              input, output, this->getSize(), comm, stream);
        },
        OpType::ALLTOALL_BASE,
        "mccl:all_to_all");
  } else {
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);

    RECORD_PARAM_COMMS_DATA(
        std::make_tuple(
            static_cast<int64_t>(seqCollective_) + 1,
            false), // seq + 1 to match collective
        std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_allv", // collective name
        inputTensor.numel(), // inNelems
        outputTensor.numel(), // outNelems
        inputTensor.scalar_type(), // dType
        inputSplitSizes, // inSplitSizes
        outputSplitSizes, // outSplitSizes
        globalRankStart, // globalRankStart
        globalRankStride, // globalRankStride
        this->getSize()); // worldSize

    // avoidRecordStreams_ note: collective() will stash inputTensors and
    // outputTensors.
    return collective(
        inputTensor,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            mcclComm_t comm,
            c10::musa::MUSAStream& stream) {
          std::vector<size_t> send_lengths(size_);
          std::vector<size_t> recv_lengths(size_);
          std::vector<size_t> send_offsets(size_);
          std::vector<size_t> recv_offsets(size_);
          c10d::computeLengthsAndOffsets(
              inputSplitSizes, input, &send_lengths, &send_offsets);
          c10d::computeLengthsAndOffsets(
              outputSplitSizes, output, &recv_lengths, &recv_offsets);
          // See [Sync Streams].
          if (!avoidRecordStreams_) {
            c10::musa::MUSACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          return all2all_single_unequal_split_impl(
              input.data_ptr(),
              send_lengths.data(),
              send_offsets.data(),
              output.data_ptr(),
              recv_lengths.data(),
              recv_offsets.data(),
              input.element_size(),
              input.scalar_type(),
              comm,
              stream);
        },
        OpType::ALLTOALL_BASE,
        "mccl:all_to_all");
  }
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& /* unused */) {
  std::vector<int64_t> inSplitSizes;
  std::vector<int64_t> outSplitSizes;
  int64_t total_numel = 0;

  auto device = outputTensors[0].device();
  for (const auto r : c10::irange(outputTensors.size())) {
    check_gpu_single_tensor(outputTensors[r]);
    check_gpu_single_tensor(inputTensors[r]);
    TORCH_CHECK(
        device == outputTensors[r].device() &&
            device == inputTensors[r].device(),
        "Tensors must be on the same device")
    inSplitSizes.push_back(inputTensors[r].numel());
    outSplitSizes.push_back(outputTensors[r].numel());
    total_numel += inputTensors[r].numel();
  }

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "all_to_all", // collective name
      total_numel, // inNelems
      total_numel, // outNelems
      inputTensors.front().scalar_type(), // dType
      inSplitSizes, // inSplitSizes
      outSplitSizes, // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  return collective(
      inputTensors,
      outputTensors,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream) {
        return all2all_impl(outputTensors, inputTensors, comm, stream);
      },
      [&](c10::musa::MUSAStream&,
          c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {
        if (avoidRecordStreams_) {
          // inputTensor0 and outputTensor0 are stashed redundantly by
          // collective(), but that's ok.
          auto& v = work->stashed_for_allocator_safety_;
          v->insert(v->end(), inputTensors.begin(), inputTensors.end());
          v->insert(v->end(), outputTensors.begin(), outputTensors.end());
        }
      },
      [](at::musa::MUSAStream&,
         c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {},
      OpType::ALLTOALL,
      "mccl:all_to_all");
}

// torch_musa P2P comms call mccl Operators directly.
// As we only use the latest mccl version and no need of
// backward compatible.
c10::intrusive_ptr<Work> ProcessGroupMCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /* unused */) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  check_gpu_single_tensor(tensor, true);

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqP2P_) + (coalescing_state_ & CoalP2P ? 0 : 1),
          true), // the 1st p2p in coalesced range sets coalescing_state_ and
                 // bumps seqP2P_
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      dstRank, // dst rank
      "send", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& input,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream,
          int dst) {
        return mcclSend(
            input.data_ptr(),
            input.numel(),
            getMcclDataType(input.scalar_type()),
            dst,
            comm,
            stream.stream());
      },
      dstRank,
      OpType::SEND,
      c10::str("mccl:send ", rank_, "->", dstRank).c_str());
  return ret;
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /* unused */) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  check_gpu_single_tensor(tensor, true);

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqP2P_) + (coalescing_state_ & CoalP2P ? 0 : 1),
          true), // the 1st p2p in coalesced range sets coalescing_state_ and
                 // bumps seqP2P_
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      srcRank, // src rank
      "recv", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& output,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream,
          int src) {
        return mcclRecv(
            output.data_ptr(),
            output.numel(),
            getMcclDataType(output.scalar_type()),
            src,
            comm,
            stream.stream());
      },
      srcRank,
      OpType::RECV,
      c10::str("mccl:recv ", rank_, "<-", srcRank).c_str());
  return ret;
}

void ProcessGroupMCCL::groupStart() {
  C10D_MCCL_CHECK(mcclGroupStart(), c10::nullopt);
  ++mcclActiveGroupCounter_;
}

void ProcessGroupMCCL::groupEnd() {
  C10D_MCCL_CHECK(mcclGroupEnd(), c10::nullopt);
  --mcclActiveGroupCounter_;
}

void ProcessGroupMCCL::groupEndNonblocking(
    const std::shared_ptr<MCCLComm>& comm) {
  if (!useNonblocking()) {
    C10D_MCCL_CHECK(mcclGroupEnd(), std::nullopt);
  } else {
    // TODO: add nonblocking mode
  }
  --mcclActiveGroupCounter_;
}

// gather impl is assembled by send & recv,
// mainly same as gather in torch/csrc/cuda/nccl.cpp
mcclResult_t gather_impl(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    mcclComm_t comm,
    c10::musa::MUSAStream& stream,
    int32_t root) {
  int numranks, cur_rank;
  C10D_MCCL_ASSERT(mcclCommCount(comm, &numranks));
  C10D_MCCL_ASSERT(mcclCommUserRank(comm, &cur_rank));

  size_t count = inputs.numel();
  auto type = getMcclDataType(inputs.scalar_type());
  const auto* sendbuff = reinterpret_cast<void*>(inputs.data_ptr());
  // TODO(yueran-tang): Maybe use AutoMcclGroup instead.
  C10D_MCCL_ASSERT(mcclGroupStart());

  if (cur_rank == root) {
    for (const auto r : c10::irange(numranks)) {
      if (r != root) {
        auto* recvbuff = reinterpret_cast<void*>(outputs[r].data_ptr());
        C10D_MCCL_ASSERT(mcclRecv(recvbuff, count, type, r, comm, stream));
      } else {
        // on its own rank, simply copy from the input
        outputs[r].copy_(inputs);
      }
    }
  } else {
    C10D_MCCL_ASSERT(mcclSend(sendbuff, count, type, root, comm, stream));
  }
  C10D_MCCL_ASSERT(mcclGroupEnd());
  return mcclSuccess;
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupMCCL::gather: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);

  TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto inputTensor = inputTensors.back();

  std::vector<at::Tensor> outputs;

  if (getRank() == opts.rootRank) {
    if (outputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element output list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (outputTensors[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect output list size " << outputTensors[0].size()
         << ". Output list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = inputTensor.options();
    const auto& sizes = inputTensor.sizes();
    assertTypeAndSizesMatch(invalidArgument, outputTensors[0], options, sizes);
    outputs = outputTensors[0];
  } else {
    // if not in the root rank, initialize outputs as empty list
    if (!outputTensors.empty()) {
      invalidArgument("requires empty output on non-root");
    }
    outputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    outputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      opts.rootRank, // root rank
      "gather", // collective name
      inputTensor.numel(), // inNelems
      inputTensor.numel() * this->getSize(), // outNelems
      inputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash inputTensors and
  // outputs, which == outputTensors[0] on the root rank where it matters.

  auto inputs = std::vector<at::Tensor>{inputTensor};

  return collective(
      inputs,
      outputs,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          if (!avoidRecordStreams_) {
            for (auto output : outputs) {
              c10::musa::MUSACachingAllocator::recordStream(
                  output.storage().data_ptr(), stream);
            }
          }
        }
        return gather_impl(
            inputTensor, outputs, comm, stream, static_cast<int32_t>(root));
      },
      [](at::musa::MUSAStream&,
         c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {},
      [](at::musa::MUSAStream&,
         c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {},
      OpType::GATHER,
      "mccl:gather");
}

mcclResult_t scatter_impl(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    mcclComm_t comm,
    c10::musa::MUSAStream& stream,
    int32_t root) {
  int numranks, cur_rank;
  C10D_MCCL_ASSERT(mcclCommCount(comm, &numranks));
  C10D_MCCL_ASSERT(mcclCommUserRank(comm, &cur_rank));
  // TODO(yueran-tang): Maybe use autoMcclGroup instead.
  C10D_MCCL_ASSERT(mcclGroupStart());
  if (cur_rank == root) {
    for (const auto r : c10::irange(numranks)) {
      if (r != root) {
        size_t send_count = inputs[r].numel();
        auto send_type = getMcclDataType(inputs[r].scalar_type());
        const auto* sendbuff = reinterpret_cast<void*>(inputs[r].data_ptr());
        C10D_MCCL_ASSERT(
            mcclSend(sendbuff, send_count, send_type, r, comm, stream));
      } else {
        // on its own rank, simply copy it to the output
        outputs.copy_(inputs[r]);
      }
    }
  } else {
    size_t recv_count = outputs.numel();
    auto recv_type = getMcclDataType(outputs.scalar_type());
    auto* recvbuff = reinterpret_cast<void*>(outputs.data_ptr());
    C10D_MCCL_ASSERT(
        mcclRecv(recvbuff, recv_count, recv_type, root, comm, stream));
  }
  C10D_MCCL_ASSERT(mcclGroupEnd());
  return mcclSuccess;
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupMCCL::scatter: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);

  // @lint-ignore CLANGTIDY
  auto outputTensor = outputTensors.back();

  std::vector<at::Tensor> inputs;

  if (getRank() == opts.rootRank) {
    if (inputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element input list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (inputTensors[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect input list size " << inputTensors[0].size()
         << ". Input list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = outputTensor.options();
    const auto& sizes = outputTensor.sizes();
    assertTypeAndSizesMatch(invalidArgument, inputTensors[0], options, sizes);
    inputs = inputTensors[0];
  } else {
    // if not in the root rank, initialize inputTensors as empty place holder
    // with an empty list
    if (!inputTensors.empty()) {
      invalidArgument("requires empty input on non-root");
    }
    inputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    inputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      opts.rootRank, // root rank
      "scatter", // collective name
      outputTensor.numel() * this->getSize(), // inNelems
      outputTensor.numel(), // outNelems
      outputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash outputTensors and
  // inputs, which == inputTensors[0] on the root rank where it matters.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  const auto root = opts.rootRank;
  bool nanCheck = (rank_ == root);

  auto outputs = std::vector<at::Tensor>{outputTensor};
  return collective(
      outputs,
      inputs,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          if (!avoidRecordStreams) {
            for (auto input : inputs) {
              c10::musa::MUSACachingAllocator::recordStream(
                  input.storage().data_ptr(), stream);
            }
          }
        }
        return scatter_impl(
            inputs, outputTensor, comm, stream, static_cast<int32_t>(root));
      },
      [](at::musa::MUSAStream&,
         c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {},
      [](at::musa::MUSAStream&,
         c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {},
      OpType::SCATTER,
      "mccl:scatter",
      avoidRecordStreams,
      nanCheck);
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  TORCH_CHECK(false, "ProcessGroupMCCL does not support recvAnysource");
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts) {
  check_gpu_single_tensor(input_tensor);
  check_gpu_single_tensor(output_tensor);

  if (input_tensor.dtype() != output_tensor.dtype()) {
    TORCH_CHECK(false, "output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * size_ != output_tensor.numel()) {
    TORCH_CHECK(
        false,
        "output tensor size must be equal to world_size times input tensor size");
  }

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      input_tensor, // inputTensors
      output_tensor, // outputTensors
      rank_, // rank
      "_allgather_base", // collective name
      input_tensor.numel(), // inNelems
      output_tensor.numel(), // outNelems
      output_tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash inputs and outputs.
  // Note 2: for asyncOp = false, we don't want to record streams because we
  // know that the MCCL stream will join back to the "current" stream right
  // after this op. So we might just as well keep the stream ownership of the
  // input/output tensors unchanged. The benefit would be that the
  // allocation/free of the tensors would look deterministic to the "current"
  // stream so that the caching allocator can reuse memory pool for this stream
  // in a clever way. This setting is added for libraries like FSDP which uses
  // `all_gather_into_tensor`.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      input_tensor,
      output_tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream) {
        if (!avoidRecordStreams) {
          c10::musa::MUSACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        return mcclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getMcclDataType(input.scalar_type()),
            comm,
            stream.stream());
      },
      OpType::_ALLGATHER_BASE,
      "mccl:_all_gather_base",
      avoidRecordStreams);
}

c10::intrusive_ptr<Backend> ProcessGroupMCCL::MCCLcreator(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    std::chrono::milliseconds op_time_out) {
  c10::intrusive_ptr<Options> options = Options::create();
  options->timeout = op_time_out;

  return c10::make_intrusive<ProcessGroupMCCL>(store, rank, size, options);
}

} // namespace c10d
