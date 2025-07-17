#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <iostream>
#include <thread>
#include <tuple>

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

static std::unordered_map<std::shared_ptr<MCCLComm>, int> mcclCommDevIdxMap;
static std::mutex mcclCommDevIdxMapMutex;

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

// TODO(yueran-tang): Not finished since we only support a few Ops.
mcclRedOp_t getMcclReduceOp(
    const ReduceOp& reduce_op,
    at::Tensor& input,
    const mcclDataType_t& data_type,
    const mcclComm_t& comm,
    int dev_in_group) {
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

std::string getKeyFromDevices(const std::vector<at::Device>& devices) {
  std::string deviceList;
  for (auto& device : devices) {
    if (deviceList.empty()) {
      deviceList = std::to_string(device.index());
    } else {
      deviceList += "," + std::to_string(device.index());
    }
  }
  return deviceList;
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

std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    // tensors must all be on the same device, or all on distinct devices.
    // The line below assumes that constraint has already been enforced
    // (by check_gpu_tensors_same_device or
    // check_gpu_tensors_different_devices).
    if (res.size() == 0 || tensor.device() != res[0]) {
      res.push_back(tensor.device());
    }
  }
  return res;
}

at::Device getDeviceForRank(int rank) {
  TORCH_CHECK(rank >= 0, "Invalid rank ", rank);
  auto numGPUs = c10::musa::device_count();
  int16_t deviceIdx = static_cast<int16_t>(rank % numGPUs);
  return at::Device(at::musa::kMUSA, deviceIdx);
}

void syncStreams(
    const std::vector<at::Device>& devices,
    std::vector<at::musa::MUSAEvent>& mcclEvents,
    std::vector<c10::musa::MUSAStream>& mcclStreams) {
  for (const auto i : c10::irange(devices.size())) {
    c10::musa::MUSAStream& mcclStream = mcclStreams[i];
    at::musa::MUSAEvent& mcclEvent = mcclEvents[i];
    mcclEvent.record(c10::musa::getCurrentMUSAStream(devices[i].index()));
    mcclEvent.block(mcclStream);
  }
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

std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
getMCCLCommDumpMap() {
  std::unordered_map<
      std::string /* mcclUniqueID */,
      std::unordered_map<std::string, std::string> /* dump from this comm */>
      mcclDumpMap;
  // dumpMCCLTrace is only called from the default PG (local_id_=0), but we
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

inline int computeDeltaMS(
    std::chrono::time_point<std::chrono::steady_clock> start,
    std::chrono::time_point<std::chrono::steady_clock> end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

std::string logPrefix() {
  // TODO(jihong.zhong): move to ProcessGroupMCCL class
  return "ProcessGroupMCCL LOG";
}

} // namespace

std::atomic<bool> ProcessGroupMCCL::shouldDump_(false);

std::ostream& operator<<(
    std::ostream& output,
    const ProcessGroupMCCL::WorkMCCL& workMCCL) {
  std::string workInfo;
  if (workMCCL.outputs_) {
    workInfo = c10::str(
        "WorkMCCL(",
        "SeqNum=",
        workMCCL.seq_,
        ", OpType=",
        opTypeToString(workMCCL.opType_),
        ", TensorShape=",
        (*workMCCL.outputs_)[0].sizes(),
        ", Timeout(ms)=",
        workMCCL.opTimeout_.count(),
        ")");
  } else {
    workInfo = c10::str(
        "WorkMCCL(",
        "SeqNum=",
        workMCCL.seq_,
        ", OpType=",
        opTypeToString(workMCCL.opType_),
        ", Timeout(ms)=",
        workMCCL.opTimeout_.count(),
        ")");
  }
  return output << workInfo;
}

ProcessGroupMCCL::WorkMCCL::WorkMCCL(
    const std::vector<at::Device>& devices,
    int rank,
    c10d::OpType opType,
    uint64_t seq,
    const char* profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputs,
    bool desyncDebug)
    : Work(rank, opType, profilingTitle, inputs),
      devices_(devices),
      workStartTime_(std::chrono::steady_clock::now()),
      seq_(seq) {
  // Creates the MUSA event wrappers
  // Note: The actual events are lazily created when first recorded to with
  // DEFAULT_FLAGS = musaEventDisableTiming.
  if (desyncDebug) {
    mcclStartEvents_ =
        std::make_shared<std::vector<at::musa::MUSAEvent>>(devices.size());
  }
  mcclEndEvents_ =
      std::make_shared<std::vector<at::musa::MUSAEvent>>(devices.size());
  mcclComms_.resize(devices.size());
}

ProcessGroupMCCL::WorkMCCL::WorkMCCL(const WorkMCCL& w)
    : Work(w.rank_, w.opType_),
      std::enable_shared_from_this<WorkMCCL>(w),
      devices_(w.devices_),
      mcclStartEvents_(w.mcclStartEvents_),
      mcclEndEvents_(w.mcclEndEvents_),
      mcclComms_(w.mcclComms_),
      blockingWait_(w.blockingWait_),
      opTimeout_(w.opTimeout_),
      workStartTime_(w.workStartTime_),
      seq_(w.seq_),
      startTraceUpdated_(w.startTraceUpdated_),
      store_(w.store_) {
  exception_ = w.exception_;
}

ProcessGroupMCCL::WorkMCCL::~WorkMCCL() = default;

bool ProcessGroupMCCL::WorkMCCL::isCompleted() {
  checkAndSetException();
  return exception() || finishedGPUExecutionInternal();
}

bool ProcessGroupMCCL::WorkMCCL::isStarted() {
  checkAndSetException();
  return exception() || startedGPUExecutionInternal();
}

bool ProcessGroupMCCL::WorkMCCL::isSuccess() const {
  if (exception()) {
    // Already detected an exception.
    return false;
  }

  return !checkForMCCLErrors(mcclComms_) && finishedGPUExecutionInternal();
}

void ProcessGroupMCCL::WorkMCCL::checkAndSetException() {
  if (exception()) {
    // We already have an exception.
    return;
  }

  auto exception_ptr = checkForMCCLErrors(mcclComms_);
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
  if (exception_) {
    LOG(INFO) << "[Rank " << rank_ << "]"
              << " found async exception when checking for MCCL errors: "
              << getExceptionMsgFromExceptionPtr(exception_);
  }
}

void ProcessGroupMCCL::WorkMCCL::setException(
    std::exception_ptr exception_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
}

// Helper that checks if the MCCL kernels are completed on the GPUs
bool ProcessGroupMCCL::WorkMCCL::finishedGPUExecution() {
  checkAndSetException();
  return finishedGPUExecutionInternal();
}

bool ProcessGroupMCCL::WorkMCCL::startedGPUExecutionInternal() const {
  for (const auto i : c10::irange(devices_.size())) {
    // Checking the work's corresponding MUSA events' status
    if (!(*mcclStartEvents_)[i].query()) {
      return false;
    }
  }
  return true;
}

bool ProcessGroupMCCL::WorkMCCL::finishedGPUExecutionInternal() const {
  try {
    for (const auto i : c10::irange(devices_.size())) {
      // Checking the work's corresponding MUSA events' status
      if (!(*mcclEndEvents_)[i].query()) {
        return false;
      }
    }
  } catch (const std::exception& e) {
    if (std::string(e.what()).find("driver shutting down") ==
        std::string::npos) {
      throw;
    }
    LOG(INFO) << "[Rank " << rank_
              << "] Event query failed with exception: " << e.what();
  }
  return true;
}

void ProcessGroupMCCL::WorkMCCL::checkAndThrowException() {
  // Set the appropriate exception if found.
  checkAndSetException();

  // Throw an exception, only if we have a valid exception.
  if (exception()) {
    std::rethrow_exception(exception());
  }
}

void ProcessGroupMCCL::WorkMCCL::handleMCCLGuard(
    ErrorHandlingMode asyncErrorHandling) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (exception_) {
    auto exceptionMsg = c10::str(
        "Some MCCL operations have failed or timed out. Due to the ",
        "asynchronous nature of MUSA kernels, subsequent GPU operations ",
        "might run on corrupted/incomplete data.");
    LOG(ERROR) << exceptionMsg;
    C10_LOG_API_USAGE_ONCE("ProcessGroupMCCL.WorkMCCL.handleMCCLGuard");
    if (asyncErrorHandling == TearDown) {
      auto tearDownMsg = c10::str(
          "To avoid data inconsistency, we are taking the entire process down.");
      LOG(ERROR) << tearDownMsg;
      std::rethrow_exception(exception_);
    }
  }
}

void ProcessGroupMCCL::WorkMCCL::synchronize() {
  // Call Synchronize without a timeout. We use this method to avoid adding a
  // timeout argument to the public synchronize API.
  synchronizeInternal(kNoTimeout);
}

void ProcessGroupMCCL::WorkMCCL::synchronizeStreams() {
  for (const auto i : c10::irange(devices_.size())) {
    auto currentStream = c10::musa::getCurrentMUSAStream(devices_[i].index());
    // Block the current stream on the MCCL stream
    (*mcclEndEvents_)[i].block(currentStream);
  }
  if (avoidRecordStreams_) {
    stashed_for_allocator_safety_->clear();
  }
}

// Waiting on the work's corresponding MUSA events
void ProcessGroupMCCL::WorkMCCL::synchronizeInternal(
    std::chrono::milliseconds timeout) {
  synchronizeStreams();

  // In case of blocking, wait for the operation to complete.
  if (blockingWait_) {
    // Wait for the operation to complete.
    while (!isCompleted()) {
      if (timedOut()) {
        // When operation times out due to some errors that are not
        // detected by mccl communicators, mcclCommWatchdog can not check this
        // time out error and thus can not abort mcclComms accordingly.
        // So explicitly abort mcclComms here before throwing this timed out
        // exception to users, after this, mcclCommWatchdog can detect mccl
        // communicators are aborted and clean up devMCCLCommMap_ accordingly.
        // if throwing timed out excepiton without aborting mccl communicators
        // here, it was observed that MUSA GPU will have 100% utilization and
        // can not run new events successfully.

        std::stringstream ss;
        ss << *this;
        auto timeoutErrorMsg =
            c10::str("Work ", ss.str(), " timed out in call to wait().");
        for (const auto& mcclComm : mcclComms_) {
          mcclComm->mcclCommAbort(timeoutErrorMsg);
          const auto& storeKey = getMcclAbortedCommStoreKey(
              buildMcclUniqueIdStr(mcclComm->getMcclId()));
          auto rankStr = std::to_string(rank_);
          store_->set(storeKey, rankStr);
          LOG(INFO) << "[Rank " << rank_
                    << "] Wrote aborted communicator id to store: " << storeKey;
        }
        auto currentTimepoint = std::chrono::steady_clock::now();
        auto timeElapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                currentTimepoint - workStartTime_);
        std::string exceptionMsg = c10::str(
            "[Rank ",
            rank_,
            "] ",
            "Caught collective operation timeout: ",
            (*this),
            " ran for ",
            timeElapsed.count(),
            " milliseconds before timing out.");
        TORCH_CHECK(false, exceptionMsg);
      }
      // Check for errors and throw appropriate exception.
      checkAndThrowException();
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
    checkAndThrowException();
  }

  // Device synchronize only after we've completed timeout checks.
  if (!barrierTensors_.empty()) {
    // If we use the work to do barrier, we should block here
    for (auto& device : devices_) {
      c10::musa::MUSAGuard gpuGuard(device);
      musaDeviceSynchronize();
    }
  }
}

// Same as calling synchronize().
bool ProcessGroupMCCL::WorkMCCL::wait(std::chrono::milliseconds timeout) {
  RECORD_PARAM_COMMS(
      static_cast<int>(this->seq_), // seq
      std::make_tuple(
          std::string("0"), std::string("wait")), // process group name
      rank_, // rank
      "wait", // colName
      0, // inSize
      0, // outSize
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      0,
      0,
      static_cast<int>(devices_.size()));
  synchronizeInternal(timeout);
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupMCCL::WorkMCCL::abort() {
  TORCH_CHECK(false, "ProcessGroupMCCL::WorkMCCL::abort not implemented.");
}

bool ProcessGroupMCCL::WorkMCCL::timedOut() {
  auto currentTimepoint = std::chrono::steady_clock::now();
  return (
      std::chrono::duration_cast<std::chrono::milliseconds>(
          currentTimepoint - workStartTime_) >= opTimeout_);
}

ProcessGroupMCCL::CoalescedWorkMCCL::CoalescedWorkMCCL(
    std::vector<ProcessGroupMCCL::WorkMCCL> works,
    int rank,
    OpType opType)
    : Work(rank, opType, nullptr), works_(std::move(works)) {}

ProcessGroupMCCL::CoalescedWorkMCCL::~CoalescedWorkMCCL() = default;

c10::intrusive_ptr<ProcessGroupMCCL::CoalescedWorkMCCL> ProcessGroupMCCL::
    initCoalescedWork(
        const std::vector<c10::intrusive_ptr<Work>>& works,
        int rank,
        OpType opType) {
  std::vector<ProcessGroupMCCL::WorkMCCL> mcclWorks;
  mcclWorks.reserve(works.size());
  for (auto& work : works) {
    mcclWorks.push_back(*static_cast<ProcessGroupMCCL::WorkMCCL*>(work.get()));
  }
  return c10::make_intrusive<ProcessGroupMCCL::CoalescedWorkMCCL>(
      mcclWorks, rank, opType);
}

// Same as calling synchronize().
bool ProcessGroupMCCL::CoalescedWorkMCCL::wait(
    std::chrono::milliseconds timeout) {
  for (auto& w : works_) {
    w.wait(timeout);
  }
  // Always return true, because abort API is not implemented.
  return true;
}

ProcessGroupMCCL::ProcessGroupMCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(options),
      mcclCommCounter_(0),
      traceKeyStart_(getTraceStartKey("MCCL", rank)),
      traceKeyEnd_(getTraceEndKey("MCCL", rank)),
      terminateProcessGroup_(false),
      terminateHeartbeatMonitorThread_(false) {
  TORCH_CHECK(
      c10::musa::device_count() != 0,
      "ProcessGroupMCCL is only supported with GPUs, no GPUs found!");
  blockingWait_ = getCvarBool(TORCH_MCCL_BLOCKING_WAIT, false);
  asyncErrorHandling_ = static_cast<ErrorHandlingMode>(
      getCvarInt(TORCH_MCCL_ASYNC_ERROR_HANDLING, 3 /*SkipCleanUp*/));
  desyncDebug_ = getCvarBool(TORCH_MCCL_DESYNC_DEBUG, false) ||
      (dist_debug_level_ >= DebugLevel::Detail);

  avoidRecordStreams_ = getCvarBool(TORCH_MCCL_AVOID_RECORD_STREAMS, false);
  coordCheckIntervalMilSec_ = getCvarInt(TORCH_MCCL_COORD_CHECK_MILSEC, 1000);
  mcclHeartBeatMonitor_ = getCvarBool(TORCH_MCCL_HEARBEAT_MONITOR, false);
  waitTimeoutDumpInMilSec_ =
      getCvarInt(TORCH_MCCL_WAIT_TIMEOUT_DUMP_MILSEC, 60 * 1000 /*60 Sec*/);
  heartbeat_ = 1ULL;

  if (blockingWait_) {
    if (asyncErrorHandling_ != NoHandling || desyncDebug_) {
      LOG(INFO) << "[Rank " << rank_ << "] MCCL_BLOCKING_WAIT and "
                << "MCCL_ASYNC_ERROR_HANDLING|MCCL_DESYNC_DEBUG"
                << "should not both be enabled. "
                << "Only MCCL_BLOCKING_WAIT is being used in this process.";
      asyncErrorHandling_ = NoHandling;
      desyncDebug_ = false;
    }
  } else {
    if (desyncDebug_ && asyncErrorHandling_ == NoHandling) {
      LOG(INFO) << "[Rank " << rank_
                << "] MCCL_DESYNC_DEBUG and MCCL_ASYNC_ERROR_HANDLING "
                << "must both be enabled. "
                << "Enabling MCCL_ASYNC_ERROR_HANDLING.";
      asyncErrorHandling_ = TearDown;
    }
  }

  if (getCvarBool(ENABLE_MCCL_HEALTH_CHECK, false)) {
    // Perform health check by initializing dummy communicators and destroying
    // them. This will help indicate any MCCL-related issues prior to the first
    // collective.
    // Run it in a separate thread and wait on CV to handle timeouts, since
    // majority of getMCCLComm failures are hangs.
    runHealthCheck();
  }

  mcclCommWatchdogThread_ =
      std::thread(&ProcessGroupMCCL::mcclCommWatchdog, this);

  if (asyncErrorHandling_ != NoHandling) {
    workCleanupThread_ = std::thread(&ProcessGroupMCCL::workCleanupLoop, this);
  }

  init();
  LOG(INFO) << "[Rank " << rank_
            << "] ProcessGroupMCCL initialized with following options:"
            << "\nMCCL_ASYNC_ERROR_HANDLING: " << asyncErrorHandling_
            << "\nMCCL_DESYNC_DEBUG: " << desyncDebug_
            << "\nMCCL_BLOCKING_WAIT: " << blockingWait_
            << "\nTIMEOUT(ms): " << options_->timeout.count()
            << "\nUSE_HIGH_PRIORITY_STREAM: "
            << options_->is_high_priority_stream;

  RECORD_PARAM_COMMS(
      0, // seq
      std::make_tuple(getBackendName(), std::string()), // process group name
      rank, // rank
      "init", // colName
      0, // inSize
      0, // outSize
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      0,
      0,
      size_);
}

void ProcessGroupMCCL::runHealthCheck() {
  // Run health check in a separate thread and wait on CV to handle timeouts,
  // since majority of getMCCLComm failures are hangs.

  struct HealthCheckData {
    std::mutex healthCheckMutex;
    std::condition_variable healthCheckCv;
    bool healthCheckSuccess = false;
    std::exception_ptr healthCheckException;
  };

  HealthCheckData healthCheckData;
  auto t = std::thread([&healthCheckData, this]() {
    try {
      std::vector<at::Device> rankDevice = {getDeviceForRank(rank_)};
      const auto key = getKeyFromDevices(rankDevice);
      // OpType does not matter, only need to set to not go through send/recv
      // path.
      getMCCLComm(key, rankDevice, OpType::ALLREDUCE);
      // Now destroy the communicators and remove them from cache so we don't
      // use destroyed communicators.
      destroyMCCLComms(key);
      // Notify main thread the health check is complete.
      {
        std::lock_guard<std::mutex> lk(healthCheckData.healthCheckMutex);
        healthCheckData.healthCheckSuccess = true;
      }
      healthCheckData.healthCheckCv.notify_one();
    } catch (const std::exception& e) {
      // Populate exception ptr.
      healthCheckData.healthCheckException = std::current_exception();
      // Unblock waiting main thread which will report exception.
      healthCheckData.healthCheckCv.notify_one();
    } // Unknown exceptions will just cause the program to terminate.
  });
  // We don't need to join the thread, just need to verify health check via the
  // CV. Hence we detach the thread here.
  t.detach(); // NOLINT
  LOG(INFO) << "[Rank " << rank_ << "]"
            << " will wait up to " << options_->timeout.count()
            << " msec for MCCL health check to complete.";
  std::unique_lock<std::mutex> lock(healthCheckData.healthCheckMutex);
  healthCheckData.healthCheckCv.wait_for(
      lock, options_->timeout, [&healthCheckData]() {
        return healthCheckData.healthCheckSuccess;
      });

  if (healthCheckData.healthCheckException) {
    std::rethrow_exception(healthCheckData.healthCheckException);
  }
  // If there is no exception, the likely culprit is a timeout/hang which is how
  // most communicator init issues manifest themselves.
  TORCH_CHECK(
      healthCheckData.healthCheckSuccess,
      "ProcessGroupMCCL: Health check failure: Failed to initialize MCCL communicator on rank ",
      rank_);
}

std::string ProcessGroupMCCL::dumpMCCLTrace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  auto mcclDumpMap = getMCCLCommDumpMap();
  return MCCLTraceBuffer::get()->dump(
      mcclDumpMap, includeCollectives, includeStackTraces, onlyActive);
}

bool ProcessGroupMCCL::dumpDebuggingInfo() {
  // Serialize all calls to this function to avoid corrupting data, but allow
  // multiple calls in one runtime. User is responsible for preserving the
  // output file from an earlier call before a later call overwrites it.
  static std::mutex writeDebugInfoMutex;
  std::lock_guard<std::mutex> lock(writeDebugInfoMutex);
  LOG(ERROR) << "ProcessGroupMCCL preparing to dump debug info.";
  if (mcclTraceBufferSize_ > 0) {
    // We dump mccl trace into local disk by default and users can register
    // their customized writer by inheriting `DebugInfoWriter` via
    // `registerDebugInfoWriter`.
    auto mcclTrace = dumpMCCLTrace(true, true, false);
    DebugInfoWriter& writer = DebugInfoWriter::getWriter(rank_);
    LOG(INFO) << "ProcessGroupMCCL dumping mccl trace to "
              << writer.getWriterTarget();
    writer.write(mcclTrace);
    return true;
  }
  return false;
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

void ProcessGroupMCCL::heartbeatMonitor() {
  // c10::setThreadName("pt_mccl_heartbt");

  uint64_t heartBeatCounter = 0ULL;
  std::string errorMsg;
  std::string exitReason;
  bool checkDumpSignal = (dumpOnTimeoutOrEx_ && local_id_ == 0);
  int monitorPollInterval =
      kHeartBeatThreadSleepMillis * 5; // the poll interval is 10s

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
        // Wrap globalStore_->check() in a try-catch block to avoid crashing if
        // the store is not available.
        bool checkExceptionDump = false;
        try {
          checkExceptionDump = store_->check({std::string(EXCEPTION_DUMP)});
        } catch (const std::exception& e) {
          LOG(WARNING)
              << logPrefix()
              << "Failed to check the \"should dump\" flag on TCPStore, "
              << "(maybe TCPStore server has shut down too early), with error: "
              << e.what();
          // We give up for now assuming TCPStore has been torn down.
          return;
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
            auto vec = store_->get(std::string(EXCEPTION_DUMP));
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
        heartbeatTimeoutInSec_ * 1000) {
      // Check the heart beat of watchdog thread.
      lastTimeHeartBeatCheck = currentTime;
      auto heartbeat = heartbeat_.load();
      if (heartbeat != heartBeatCounter) {
        heartBeatCounter = heartbeat;
      } else {
        shouldDump_.store(true);
        // Watchdog heartbeat timeout.
        errorMsg = c10::str(
            "heartbeatMonitor", /*logPrefix(),*/
            "ProcessGroupMCCL's watchdog got stuck for ",
            heartbeatTimeoutInSec_,
            " seconds without making progress in monitoring enqueued collectives. ",
            "This typically indicates a MCCL/CUDA API (e.g., CudaEventDestroy) hang blocking the watchdog, ",
            "and could be triggered by another thread holding the GIL inside a ",
            "CUDA api (for example, CudaEventDestroy), or other deadlock-prone behaviors.",
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

  if (checkDumpSignal && shouldDump_.load()) {
    // Store debug info to storage if no other thread does it. (By default to
    // local disk)
    std::future<bool> asyncDebugDump = std::async(
        std::launch::async, [this]() { return this->dumpDebuggingInfo(); });

    // wait for the dump until timeout - log data
    waitForFutureOrTimeout(
        asyncDebugDump,
        std::chrono::milliseconds(waitTimeoutDumpInMilSec_),
        "Flight recorder dump in heartbeatMonitor",
        false,
        true);
  }
}

void ProcessGroupMCCL::waitForFutureOrTimeout(
    std::future<bool>& fut,
    const std::chrono::milliseconds& timeOutMilSec,
    const std::string& futDescription,
    bool throwException,
    bool log) {
  std::string errorMsg;

  //   data.strings["flight_recorder_version"] = c10d::version_val_str;
  // }

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
      }
    } catch (const std::exception& e) {
      errorMsg = c10::str(
          logPrefix(),
          "Exception thrown when waiting for future ",
          futDescription,
          ": ",
          e.what());
      LOG(ERROR) << errorMsg;
    } catch (...) {
      errorMsg = c10::str(
          logPrefix(),
          "Unknown exception thrown when waiting for future ",
          futDescription);
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
}

void ProcessGroupMCCL::setSequenceNumberForGroup() {}

uint64_t ProcessGroupMCCL::getSequenceNumberForGroup() {
  return seq_;
}

ProcessGroupMCCL::~ProcessGroupMCCL() {
  terminateProcessGroup_.store(true);
  // We need to wait for abort to finish before we can safely shut down
  // heartbeat monitoring thread.
  terminateHeartbeatMonitorThread_.store(true);

  watchdogCV_.notify_one();
  mcclCommWatchdogThread_.join();

  if (mcclHeartBeatMonitor_ && mcclHeartbeatMonitorThread_.joinable()) {
    mcclHeartbeatMonitorThread_.join();
  }
  if (asyncErrorHandling_ != NoHandling) {
    workMetaListCV_.notify_one();
    workCleanupThread_.join();
  }

  {
    // Abort all MCCL Communicators on Process Group Destruction
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& it : devMCCLCommMap_) {
      auto& mcclComms = it.second;

      for (const auto& mcclComm : mcclComms) {
        std::string abortReason =
            c10::str("Process Group destroyed on rank ", rank_);
        mcclComm->mcclCommAbort(abortReason);
      }
    }
  }
}

void ProcessGroupMCCL::abortTimedOutCollectives(
    std::unordered_set<std::string>& abortedCommIds) {
  std::unique_lock<std::mutex> lock(workMetaListMutex_);
  for (auto& work : workMetaList_) {
    work.checkAndSetException();
    // Aborting MCCL Communicators due to errors is already handled above.
    if (work.exception()) {
      auto vec = std::vector<uint8_t>(
          reinterpret_cast<uint8_t*>(const_cast<int*>(&rank_)),
          reinterpret_cast<uint8_t*>(const_cast<int*>(&rank_)) + sizeof(rank_));
      store_->set(std::string(EXCEPTION_DUMP), vec);

      if (!shouldDump_.load()) {
        LOG(ERROR)
            << logPrefix()
            << "Broadcasting flight-recorder dump signal to other processes via TCPStore.";
      }
      // signal the monitor thread on PG0 to start dumping
      shouldDump_.store(true);

      continue;
    }

    // Check for Timeouts in the WorkMCCL Operations, and abort all
    // communicators accordingly.
    if (work.timedOut()) {
      auto currentTimepoint = std::chrono::steady_clock::now();
      auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          currentTimepoint - work.workStartTime_);
      std::string exceptionMsg = c10::str(
          "[Rank ",
          rank_,
          "] ",
          "Watchdog caught collective operation timeout: ",
          work,
          " ran for ",
          timeElapsed.count(),
          " milliseconds before timing out.");
      if (desyncDebug_) {
        exceptionMsg += retrieveDesyncReport(store_, "MCCL", rank_, size_);
      }
      LOG(ERROR) << exceptionMsg;
      std::exception_ptr exception_ptr =
          std::make_exception_ptr(std::runtime_error(exceptionMsg));
      work.setException(exception_ptr);
      for (const auto& mcclComm : work.mcclComms_) {
        mcclComm->mcclCommAbort(exceptionMsg);
        abortedCommIds.emplace(buildMcclUniqueIdStr(mcclComm->getMcclId()));
      }
    }
  }
}

void ProcessGroupMCCL::mcclCommWatchdog() {
  try {
    if (mcclHeartBeatMonitor_) {
      mcclHeartbeatMonitorThread_ =
          std::thread(&ProcessGroupMCCL::heartbeatMonitor, this);
    }

    LOG(INFO) << "[Rank " << rank_ << "] MCCL watchdog thread started!";
    mcclCommWatchdogInternal();
    LOG(INFO) << "[Rank " << rank_
              << "] MCCL watchdog thread terminated normally";
  } catch (std::exception& e) {
    LOG(INFO) << "[Rank " << rank_
              << "] MCCL watchdog thread terminated with exception: "
              << e.what();
  } catch (...) {
    LOG(INFO) << "[Rank " << rank_
              << "] MCCL watchdog thread terminated with unknown exception";
  }
}

void ProcessGroupMCCL::mcclCommWatchdogInternal() {
  lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
  while (!terminateProcessGroup_.load()) {
    std::unordered_set<std::string> abortedCommIds;
    std::unordered_set<std::string> allCommIds;

    {
      // Loop through the cache of communicators for MCCL errors.
      std::lock_guard<std::mutex> lock(mutex_);
      heartbeat_++;
      for (auto& it : devMCCLCommMap_) {
        auto& mcclComms = it.second;

        for (const auto& mcclComm : mcclComms) {
          allCommIds.emplace(buildMcclUniqueIdStr(mcclComm->getMcclId()));
        }
        std::exception_ptr mcclErrorException = checkForMCCLErrors(mcclComms);
        if (mcclErrorException) {
          auto exceptionMsg =
              getExceptionMsgFromExceptionPtr(mcclErrorException);
          LOG(INFO)
              << "[Rank " << rank_
              << "] Received MCCL errors for communicators in the cache: \n"
              << "MCCL error: \n"
              << exceptionMsg;

          if (blockingWait_ || asyncErrorHandling_ != NoHandling) {
            LOG(INFO) << "[Rank " << rank_
                      << "] Aborting communicators that received errors";
            // We abort MCCL communicators that have received errors from this
            // thread, and exceptions are set on the corresponding work objects.
            // The workCleanupThread will then loop through the unfinished
            // collectives and throw exceptions if an exception has been set on
            // any of the work objects from this thread.
            for (const auto& mcclComm : mcclComms) {
              // We are aborting remaining communicators due to an error in
              // at least one of these communicators, so propagate that reason
              // for better debugability.
              mcclComm->mcclCommAbort(exceptionMsg);
              // Note that we don't remove the aborted communicators from the
              // cache. The reason is that if we do remove the communicator
              // from the cache, it is possible that a new collective operation
              // calls `mcclCommInitRank` to create a new communicator whereas
              // other ranks might have failed/timed out and didn't enter
              // `mcclCommInitRank`. As a result, when there is a failure on
              // a communicator the application receives an exception and its
              // their responsibility to destroy the process group and recreate
              // it to recover from errors.
              abortedCommIds.emplace(
                  buildMcclUniqueIdStr(mcclComm->getMcclId()));
            }
          }
        }
      }
    }

    if (asyncErrorHandling_ != NoHandling) {
      abortTimedOutCollectives(abortedCommIds);
    }

    if (blockingWait_) {
      // When we abort a communicator on one rank, it is likely that might cause
      // other ranks to hang indefinitely. As a result, whenever we abort a
      // communicator, we write its ID to the store. The watchdog on other ranks
      // then monitor the store, find an aborted communicator ID and abort their
      // respective communicator as well.

      // Record the aborted communicators locally and in the store.
      for (const auto& abortedCommId : abortedCommIds) {
        abortedComms_.emplace(abortedCommId);
        const auto& storeKey = getMcclAbortedCommStoreKey(abortedCommId);
        auto rankStr = std::to_string(rank_);
        store_->set(storeKey, rankStr);
        LOG(INFO) << "[Rank " << rank_
                  << "] Watchdog wrote aborted communicator id to store: "
                  << storeKey;
      }

      // Check for any communicators in the store and abort them if needed.
      for (const auto& commId : allCommIds) {
        if (abortedComms_.find(commId) == abortedComms_.end()) {
          // Check if we need to abort them if not already aborted (shouldn't
          // wait more than the watchdog sleep time.).
          const auto& storeKey = getMcclAbortedCommStoreKey(commId);
          try {
            store_->wait(
                {storeKey},
                std::chrono::milliseconds(kWaitForAbortCommStoreKey));
            auto val = store_->get(storeKey);
            std::string rank(reinterpret_cast<char*>(val.data()), val.size());
            std::stringstream ss;
            ss << "[Rank " << rank_ << "] Found key in store: " << storeKey
               << ", from rank: " << rank
               << ". This means that rank has aborted its MCCL communicators previously and is not in a healthy state."
               << ". Aborting appropriate communicators";
            std::string abortReason = ss.str();
            LOG(WARNING) << abortReason;

            // Now abort the appropriate communicators.
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = mcclIdToCommMap_.find(commId);
            TORCH_INTERNAL_ASSERT(it != mcclIdToCommMap_.end());
            for (const auto& mcclComm : it->second) {
              // The reason we are aborting is because some other ranks have
              // aborted their communicators originally, so propagate that
              // reason.
              mcclComm->mcclCommAbort(abortReason);
            }
            abortedComms_.emplace(commId);
            LOG(INFO) << "[Rank " << rank_
                      << "] Aborted communicators for key in store: "
                      << storeKey;
          } catch (std::exception& e) {
            VLOG(1) << "Did not find key in store: " << storeKey
                    << ", error: " << e.what();
          }
        }
      }
    }

    std::unique_lock<std::mutex> lock(watchdogCVMutex_);
    watchdogCV_.wait_for(
        lock,
        std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool { return terminateProcessGroup_.load(); });
  }
}

void ProcessGroupMCCL::workCleanupLoop() {
  bool done = false;
  while (!terminateProcessGroup_.load() || !done) {
    std::list<WorkMCCL> doneWorks;
    {
      std::unique_lock<std::mutex> lock(workMetaListMutex_);
      // We busy-poll the work vector every kWatchdogThreadSleepMillis
      // milliseconds as long as the atomic is True.
      workMetaListCV_.wait_for(
          lock,
          std::chrono::milliseconds(kWorkCleanupThreadSleepMillis),
          [&]() -> bool { return terminateProcessGroup_.load(); });

      for (auto it = workMetaList_.begin(); it != workMetaList_.end();
           /* no increment*/) {
        auto& work = *it;

        if (desyncDebug_ && !work.exception()) {
          if (!work.startTraceUpdated_ && work.isStarted() &&
              !terminateProcessGroup_.load() && !storeError_) {
            work.startTraceUpdated_ = true;
            storeError_ = !c10d::traceUpdate(
                store_,
                traceKeyStart_,
                work.seq_,
                opTypeToString(work.opType_));
          }
        }

        if (work.isCompleted()) {
          if (desyncDebug_ && !work.exception()) {
            // To close the window between the check of work.isStarted() and
            // the check of work.isCompleted().
            if (!work.startTraceUpdated_ && !terminateProcessGroup_.load() &&
                !storeError_) {
              storeError_ = !c10d::traceUpdate(
                  store_,
                  traceKeyStart_,
                  work.seq_,
                  opTypeToString(work.opType_));
            }
            if (!terminateProcessGroup_.load() && !storeError_) {
              storeError_ = !c10d::traceUpdate(
                  store_,
                  traceKeyEnd_,
                  work.seq_,
                  opTypeToString(work.opType_));
            }
          }
          // Handle Exceptions on failed GPU operations and remove completed
          // workMCCL objects from work vector.
          if (!terminateProcessGroup_.load()) {
            work.handleMCCLGuard(asyncErrorHandling_);
          }
          doneWorks.push_back(std::move(*it));
          it = workMetaList_.erase(it);
          lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
          at::musa::MUSAGraph::dec_pending_event_queries();
        } else {
          // Increment the iterator if the current WorkMCCL object is not
          // completed.
          ++it;
        }
      }
      done = workMetaList_.empty();
    }
    doneWorks.clear();
  }
}

std::exception_ptr ProcessGroupMCCL::WorkMCCL::checkForMCCLErrors(
    const std::vector<std::shared_ptr<MCCLComm>>& mcclComms) const {
  return checkForMCCLErrorsInternal(mcclComms);
}

std::exception_ptr ProcessGroupMCCL::checkForMCCLErrors(
    const std::vector<std::shared_ptr<MCCLComm>>& mcclComms) {
  return checkForMCCLErrorsInternal(mcclComms);
}

std::exception_ptr ProcessGroupMCCL::checkForMCCLErrorsInternal(
    const std::vector<std::shared_ptr<MCCLComm>>& mcclComms) {
  for (const auto& mcclComm : mcclComms) {
    // Prioritize commFailureReason over checkForMcclError() result if
    // commFailureReason is set.
    auto commFailureReason = mcclComm->getMcclCommFailureReason();
    if (commFailureReason != c10::nullopt) {
      return std::make_exception_ptr(std::runtime_error(c10::str(
          "MCCL communicator encountered error set by ProcessGroupMCCL: ",
          *commFailureReason)));
    }
    mcclResult_t mcclAsyncErr = mcclComm->checkForMcclError();
    if (mcclAsyncErr != mcclSuccess) {
      return std::make_exception_ptr(std::runtime_error(
          "MCCL error: " + mcclGetErrorWithVersion(mcclAsyncErr) + "\n" +
          getMcclErrorDetailStr(mcclAsyncErr)));
    }
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

void ProcessGroupMCCL::destroyMCCLComms(const std::string& devMCCLCommMapKey) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (devMCCLCommMap_.find(devMCCLCommMapKey) == devMCCLCommMap_.end()) {
    TORCH_INTERNAL_ASSERT(
        false,
        "Expected to find key ",
        devMCCLCommMapKey,
        " in MCCL communicator map.");
  }
  std::vector<std::shared_ptr<MCCLComm>>& mcclComms =
      devMCCLCommMap_[devMCCLCommMapKey];
  // Loop through communicators and call mcclCommAbort.
  for (const auto& comm : mcclComms) {
    // mcclCommDestroy(comm->getMcclComm()) results in segfault when PG is being
    // destroyed, so using mcclCommAbort here.
    comm->mcclCommAbort();
  }
  // Remove communicators from the cache.
  devMCCLCommMap_.erase(devMCCLCommMapKey);
  // Clear used device indices.
  usedDeviceIdxs_.clear();
}

std::vector<std::shared_ptr<MCCLComm>>& ProcessGroupMCCL::getMCCLComm(
    const std::string& devicesKey,
    const std::vector<at::Device>& devices,
    OpType opType,
    int p2pRank,
    bool isSendRecvSelf) {
  // Sanity check
  if (devicesKey.empty()) {
    TORCH_CHECK(
        false,
        "Not able to create/get the MCCL Communicator since "
        "the GPU devices are not known");
  }

  for (auto& device : devices) {
    usedDeviceIdxs_.insert(device.index());
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (devMCCLCommMap_.find(devicesKey) != devMCCLCommMap_.end()) {
      // Reuse the cached communicator if there is one.
      return devMCCLCommMap_[devicesKey];
    }
  }

  // MCCL communicator not cached, create a new entry
  std::vector<std::shared_ptr<MCCLComm>> mcclComms;
  mcclComms.resize(devices.size());

  // Create the unique MCCL ID and broadcast it
  mcclUniqueId mcclID;

  // For batch_isend_irecv, mcclGroupStart() would be called upfront
  bool batchP2P = mcclActiveGroupCounter_ > 0;
  bool singleP2POp = isP2POp(opType, batchP2P);
  // For point-to-point communication, lower rank of the two will get unique id.
  if (rank_ == 0 || (singleP2POp && p2pRank == 0)) {
    C10D_MCCL_CHECK(mcclGetUniqueId(&mcclID), c10::nullopt);
  }

  // For point-to-point communication on the same process, don't need broadcast.
  if (!isSendRecvSelf) {
    // Broadcast so that each process can have a unique MCCL ID
    broadcastUniqueMCCLID(&mcclID, singleP2POp, devicesKey, p2pRank);
  }

  c10::musa::OptionalMUSAGuard gpuGuard;

  std::vector<c10::musa::MUSAStream> streamVal;
  streamVal.reserve(devices.size());

  // [Group Start/End Note] This is used to ensure that mccl communicator will
  // be created before communication primitives are called. Let's look at this
  // example: Using the batch_isend_irecv to send a tensor to a target process.
  // On the sender side, the corresponding underlying MCCL calls will look like
  //   mcclGroupStart() // This is in batch_isend_irecv
  //   mcclGroupStart() // This is [Note 1]
  //   mcclCommInitRank() // Inside MCCLComm::create
  //   mcclSend()
  //   mcclGroupEnd() // This is [Note 2]
  //   mcclGroupEnd() // This is in batch_isend_irecv
  // With this pattern, the mccl communicator will be created in the last
  // mcclGroupEnd which means when mcclSend is processed, the passed
  // communicator argument is NULL which will lead to runtime error. So we need
  // to "close" all active mccl groups to ensure mccl communicator is actually
  // created before encountering any communication calls. This is why we need
  // the following for loop.
  for (const auto i : c10::irange(mcclActiveGroupCounter_)) {
    (void)i;
    C10D_MCCL_CHECK(mcclGroupEnd(), c10::nullopt);
  }

  // [Note 1] Create the MCCL communicators for each GPU
  C10D_MCCL_CHECK(mcclGroupStart(), c10::nullopt);

  for (const auto i : c10::irange(devices.size())) {
    // GPU world size and GPU rank
    int numRanks, rank;

    if (!singleP2POp) {
      // Collective, all-to-all, or batch P2P
      numRanks = getSize() * devices.size();
      rank = getRank() * devices.size() + i;
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
    // Get the device index
    int deviceIndex = devices[i].index();

    gpuGuard.set_index(deviceIndex);
    mcclComms[i] = MCCLComm::create(numRanks, rank, mcclID);

    // Creates the MCCL streams
    streamVal.push_back(
        c10::musa::getStreamFromPool(options_->is_high_priority_stream));
  }

  // [Note 2 ]
  C10D_MCCL_CHECK(mcclGroupEnd(), c10::nullopt);

  // At this point MCCL should have been initialized, hence we can accurately
  // get the env value even if MCCL sets it by reading from mccl.conf file
  if (getRank() == 0) {
    LOG(INFO) << "MCCL_DEBUG: " << getCvarString({"MCCL_DEBUG"}, "N/A");
  }

  // See [Group Start/End Note]
  for (const auto i : c10::irange(mcclActiveGroupCounter_)) {
    (void)i;
    C10D_MCCL_CHECK(mcclGroupStart(), c10::nullopt);
  }

  mcclStreams_.emplace(devicesKey, std::move(streamVal));

  // Note: these events are created with the (default) musaEventDisableTiming
  // flag This flag provides the best performance when used with
  // musaStreamWaitEvent() and musaEventQuery(). Since we here don't measure the
  // performance using musaEvent, this should be set.
  mcclEvents_.emplace(
      std::piecewise_construct,
      std::make_tuple(devicesKey),
      std::make_tuple(devices.size()));

  // Hold the lock before modifying the cache.
  std::lock_guard<std::mutex> lock(mutex_);

  // Record the communicators based on mcclUniqueId.
  mcclIdToCommMap_.emplace(buildMcclUniqueIdStr(mcclID), mcclComms);

  // Move the MCCL resource to cache
  devMCCLCommMap_.emplace(devicesKey, std::move(mcclComms));
  return devMCCLCommMap_[devicesKey];
}

namespace {

// Check validity of tensor
void check_gpu_single_tensor(const at::Tensor& tensor) {
  // if (!tensor.is_cuda() || tensor.is_sparse()) {
  //   TORCH_CHECK(false, "Tensors must be CUDA and dense");
  // }
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    TORCH_CHECK(false, "Tensors must be contiguous");
  }
}

// Checks that all `tensors' have the same type and shape and reside on distinct
// GPUs.
// TODO: test_c10d_mccl.py should consider adding tests for the error conditions
// here, ie, that deliberately pass invalid tensors and check the right
// exception is thrown.
void check_gpu_tensors_different_devices(
    const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    TORCH_CHECK(false, "Tensor list must be nonempty");
  }
  if (tensors.size() > static_cast<size_t>(c10::musa::device_count())) {
    TORCH_CHECK(
        false,
        "Tensor list mustn't be larger than the number of available GPUs");
  }

  const auto& first = tensors.front();

  // Set for ensuring that tensors are on separate devices.
  std::unordered_set<decltype(first.get_device())> usedDevices;
  usedDevices.reserve(tensors.size());

  for (const auto& t : tensors) {
    // if (!t.is_cuda() || t.is_sparse()) {
    //   TORCH_CHECK(false, "Tensors must be CUDA and dense");
    // }
    TORCH_CHECK(t.numel() > 0, "Tensor must be nonempty");

    if (t.scalar_type() != first.scalar_type()) {
      TORCH_CHECK(false, "Tensors must have identical type");
    }
    if (t.sizes() != first.sizes()) {
      TORCH_CHECK(false, "Tensors must have identical size");
    }
    if (t.strides() != first.strides()) {
      TORCH_CHECK(false, "Tensors must have identical strides");
    }
    if (!t.is_contiguous(t.suggest_memory_format())) {
      TORCH_CHECK(false, "Tensors must be contiguous");
    }
    const auto inserted = usedDevices.insert(t.get_device()).second;
    if (!inserted) {
      TORCH_CHECK(false, "Tensors must be on distinct GPU devices");
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
    // if (!t.is_cuda() || t.is_sparse()) {
    //   TORCH_CHECK(false, "Tensors must be CUDA and dense");
    // }
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

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other,
    size_t world_size) {
  if (tensor_lists.size() != other.size()) {
    TORCH_CHECK(
        false,
        "Tensor list operands to scatter/gather must have the same length");
  }
  const auto num_devices = tensor_lists.size();

  std::vector<at::Tensor> flattened;
  flattened.resize(num_devices);

  for (const auto i : c10::irange(size_t{}, num_devices)) {
    if (tensor_lists[i].size() != world_size * num_devices) {
      TORCH_CHECK(
          false,
          "Tensor list input to scatter/gather must match number of collective"
          " participants");
    }

    // Only check device match for the first tensor in the list; the call to
    // newLikeFlat() below will check the rest.
    if (tensor_lists[i].front().get_device() != other[i].get_device()) {
      TORCH_CHECK(
          false,
          "Corresponding input/output tensors to scatter/gather must all reside"
          " on the same device");
    }

    for (const auto& t : tensor_lists[i]) {
      if (t.numel() != other[i].numel()) {
        TORCH_CHECK(
            false,
            "All tensor operands to scatter/gather must have the same number of elements");
      }
    }
    // Flatten the tensors (from all ranks) into a single big tensor.
    flattened[i] = newLikeFlat(tensor_lists, i);
  }
  return flattened;
}

} // namespace

static std::mutex musa_free_mutex;
struct AutoMcclGroup {
  AutoMcclGroup() {
    musa_free_mutex.lock();
    C10D_MCCL_CHECK(mcclGroupStart(), "AutoMcclGroup Start Failed");
  }
  ~AutoMcclGroup() noexcept(false) {
    C10D_MCCL_CHECK(mcclGroupEnd(), "AutoMcclGroup End Failed");
    musa_free_mutex.unlock();
  }
};

c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL> ProcessGroupMCCL::initWork(
    std::vector<at::Device> devices,
    int rank,
    OpType opType,
    const char* profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputs) {
  return c10::make_intrusive<ProcessGroupMCCL::WorkMCCL>(
      devices, rank, opType, seq_, profilingTitle, inputs, desyncDebug_);
}

std::vector<at::Tensor> ProcessGroupMCCL::WorkMCCL::result() {
  return *outputs_;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupMCCL::WorkMCCL::
    getFuture() {
  return future_;
}

void ProcessGroupMCCL::workEnqueue(
    c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL> work) {
  if (!terminateProcessGroup_.load()) {
    std::lock_guard<std::mutex> lock(workMetaListMutex_);
    // Avoid view tensors to be processed in cleanup thread.
    // View tensors' destruction invokes autograd_meta, which
    // needs to be destructed in user thread. Otherwise will
    // get deadlock. Here we enqueue work without outputs_.
    workMetaList_.emplace_back(*work);
    lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
  }
}

ProcessGroupMCCL::Options::Options(bool is_high_priority_stream)
    : Backend::Options(MCCL_BACKEND_NAME),
      is_high_priority_stream(is_high_priority_stream) {}

static constexpr int CoalActive = 0x01, CoalColl = 0x02, CoalP2P = 0x04;

void ProcessGroupMCCL::startCoalescing() {
  coalescedDevices_.clear();
  coalescedComms_.clear();
  coalescing_state_ |= CoalActive;
  groupStart();
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::endCoalescing() {
  /* TODO(tyr)
   here skiped
   "if !mccl_use_nonblocking() or coalescedComms_.size() == 0"
   as torch-musa only support non_blocking mode yet.
  */
  groupEnd();

  coalescing_state_ = 0;
  if (coalescedDevices_.size() == 0) {
    return nullptr;
  }

  auto devices = coalescedDevices_[0];

  auto work = initWork(
      devices, rank_, OpType::COALESCED, "mccl:coalesced", c10::nullopt);
  const auto key = getKeyFromDevices(devices);
  auto& mcclStreams = mcclStreams_[key];
  for (const auto i : c10::irange(devices.size())) {
    auto& devEvent = (*work->mcclEndEvents_)[i];
    devEvent.record(mcclStreams[i]);
  }

  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams_;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;
  if (avoidRecordStreams_) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>();
  }

  c10::musa::CaptureStatus capture_status =
      c10::musa::currentStreamCaptureStatusMayInitCtx();

  if ((coalescing_state_ & CoalColl) &&
      capture_status == c10::musa::CaptureStatus::None) {
    // Notify graphs before we check the capture status preemptively
    at::musa::MUSAGraph::inc_pending_event_queries();
    workEnqueue(work);
  }
  return work;
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
    bool avoidRecordStreams) {
  // Environment setting by the user may add onto collective call's option
  avoidRecordStreams |= avoidRecordStreams_;
  c10::musa::CaptureStatus capture_status =
      c10::musa::currentStreamCaptureStatusMayInitCtx();
  errorIfCapturingNonCapturableMCCL();

  // Bump collective counter
  seq_++;

  // Currently, the API permits two scenarios where inputs.size() and
  // outputs.size() are > 0.
  // 1. If the call was a _coalesced call, all inputs must be on the same
  // device.
  //    The group of mccl calls applies the collective separately to each input,
  //    but the group as a whole should be efficient, and might even execute as
  //    a single fused kernel.
  // 2. If the call was a _multigpu call, all inputs must be on different
  // devices.
  //    The mccl group applies the collective across them (eg, if the collective
  //    is an allreduce, the output on each device contains contributions summed
  //    across `inputs' tensors).
  const auto devices = getDeviceList(inputs);
  const bool inputs_same_dev = (devices.size() == 1);
  const auto key = getKeyFromDevices(devices);
  auto& mcclComms = getMCCLComm(key, devices, opType);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalColl;
    coalescedDevices_.push_back(devices);
    coalescedComms_.push_back(mcclComms);
  }

  // Used many times below, so we stash the unordered_map lookup
  auto& mcclStreams = mcclStreams_[key];

  // First let MCCL streams wait for input tensors allocation streams
  syncStreams(devices, mcclEvents_[key], mcclStreams);

  // Work itself will create the MUSA events on all GPUs of tensors
  bool can_profile = outputs.size() == 1;
  auto work = initWork(
      devices,
      rank_,
      opType,
      can_profile ? profilingTitle : nullptr,
      can_profile ? c10::optional<std::vector<at::Tensor>>(inputs)
                  : c10::nullopt);

  // Store references to outputs to be used by WorkMCCL::result and operator<<.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  if (avoidRecordStreams) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>(inputs);
  }

  c10::musa::OptionalMUSAGuard gpuGuard;

  // Start event should only be recorded before the mcclGroupStart()
  if (desyncDebug_) {
    for (const auto i : c10::irange(devices.size())) {
      c10::musa::MUSAStream& mcclStream = mcclStreams[i];
      (*work->mcclStartEvents_)[i].record(mcclStream);
    }
  }

  pre(mcclStreams, work);

  {
    AutoMcclGroup mccl_group_guard;
    for (const auto i : c10::irange(inputs.size())) {
      if (!inputs_same_dev || (inputs_same_dev && i == 0)) {
        gpuGuard.set_index(devices[i].index());
      }
      decltype(i) stream_comm_i = (inputs_same_dev ? 0 : i);
      auto& mcclStream = mcclStreams[stream_comm_i];
      auto& mcclComm = mcclComms[stream_comm_i];
      // Both `inputs' and `outputs' are created on a worker stream and used in
      // different mcclStreams.  Hence, both must record the mcclStream to
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
          TORCH_CHECK(false, "sparse tensor not fully supported");
        }
      }
      C10D_MCCL_CHECK(
          fn(inputs[i], outputs[i], mcclComm->getMcclComm(), mcclStream),
          mcclComm->getMcclCommFailureReason());
    }
  }

  post(mcclStreams);

  // End event should only be recorded after the mcclGroupEnd()
  for (const auto i : c10::irange(devices.size())) {
    c10::musa::MUSAStream& mcclStream = mcclStreams[i];
    if (!coalescing_state_) {
      (*work->mcclEndEvents_)[i].record(mcclStream);
    }
    work->mcclComms_[i] = mcclComms[i];
  }

  {
    c10::musa::MUSAMultiStreamGuard streamGuard(mcclStreams);
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks. wrapCallback() in MUSA
    // future blocks the stream this callback runs on the corresponding
    // mcclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback([work](at::ivalue::Future& /* unused */) {
        work->recordFunctionEndCallback_();
      });
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Set appropriate work parameters.
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;

  if ((!coalescing_state_ || asyncErrorHandling_ != NoHandling) &&
      capture_status == c10::musa::CaptureStatus::None) {
    // Notify graphs before we check the capture status preemptively
    at::musa::MUSAGraph::inc_pending_event_queries();
    workEnqueue(work);
  }

  // TODO(tyr): Combined asyncErrorHandling and coalescing state.
  // maybe should use && instead of ||. check it in the future.
  //
  //  if (!coalescing_state_) {
  //    workEnqueue(work);
  //  }
  //  if (asyncErrorHandling_ != NoHandling) {
  //    workEnqueue(work);
  //  }

  return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupMCCL::pointToPoint(
    std::vector<at::Tensor>& tensors,
    Fn fn,
    int peer,
    OpType opType,
    PreProcess pre,
    PostProcess post,
    const char* profilingTitle) {
  if (avoidRecordStreams_) {
    TORCH_WARN_ONCE(
        "TORCH_MCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point ",
        "collectives.");
  }
  const auto devices = getDeviceList(tensors);
  std::string key;
  int p2pRank = 0, p2pTargetRank = 0;
  bool isSendRecvSelf = false;
  // For batch_isend_irecv, mcclGroupStart() would be called upfront
  bool batchP2P = mcclActiveGroupCounter_ > 0;
  if (batchP2P) {
    // For batch P2P, we need to treat it like a collective when selecting
    // communicator, because other ranks can call into this batch other than my
    // rank and my peer
    key = getKeyFromDevices(devices);
    p2pRank = rank_;
    p2pTargetRank = peer;
  } else {
    // For single P2P, preserve the old two-rank behavior (to avoid perf diff)
    key = getKeySendRecv(rank_, peer);
    p2pRank = rank_ <= peer ? 0 : 1;
    isSendRecvSelf = rank_ == peer;
    p2pTargetRank = isSendRecvSelf ? 0 : 1 - p2pRank;
  }
  auto& mcclComms = getMCCLComm(key, devices, opType, p2pRank, isSendRecvSelf);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalP2P;
    coalescedDevices_.push_back(devices);
    coalescedComms_.push_back(mcclComms);
  }

  // First let MCCL streams wait for input tensors allocation streams
  syncStreams(devices, mcclEvents_[key], mcclStreams_[key]);

  // Work itself will create the MUSA events on all GPUs of tensors
  bool can_profile = tensors.size() == 1;
  auto work = initWork(
      devices,
      rank_,
      opType,
      can_profile ? profilingTitle : nullptr,
      can_profile ? c10::optional<std::vector<at::Tensor>>(tensors)
                  : c10::nullopt);

  // Store references to outputs to be used by WorkMCCL::result and operator<<.
  // Note that these outputs are only valid for recv(), as send() does not
  // modify the inputs but we still create these outputs for use cases such as
  // profiling.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(tensors);

  c10::musa::OptionalMUSAGuard gpuGuard;

  // Start event should only be recorded before the mcclGroupStart()
  if (desyncDebug_) {
    for (const auto i : c10::irange(tensors.size())) {
      c10::musa::MUSAStream& mcclStream = mcclStreams_[key][i];
      (*work->mcclStartEvents_)[i].record(mcclStream);
    }
  }

  pre(mcclStreams_[key], work);

  for (const auto i : c10::irange(tensors.size())) {
    gpuGuard.set_index(devices[i].index());
    c10::musa::MUSAStream& mcclStream = mcclStreams_[key][i];

    // Both send tensor and recv tensor are created on a worker stream and used
    // in different mcclStreams.  Hence, both must record the mcclStream to
    // prevent being freed before the collective finishes.
    //
    // See [Sync Streams].
    c10::musa::MUSACachingAllocator::recordStream(
        tensors[i].storage().data_ptr(), mcclStream);
  }

  {
    AutoMcclGroup mccl_group_guard;
    for (const auto i : c10::irange(tensors.size())) {
      gpuGuard.set_index(devices[i].index());
      c10::musa::MUSAStream& mcclStream = mcclStreams_[key][i];
      C10D_MCCL_CHECK(
          fn(tensors[i],
             mcclComms[i]->getMcclComm(),
             mcclStream,
             p2pTargetRank),
          mcclComms[i]->getMcclCommFailureReason());
    }
  }

  post(mcclStreams_[key]);

  // End event should only be recorded after the mcclGroupEnd()
  for (const auto i : c10::irange(tensors.size())) {
    c10::musa::MUSAStream& mcclStream = mcclStreams_[key][i];
    if (!coalescing_state_) {
      (*work->mcclEndEvents_)[i].record(mcclStream);
    }
    work->mcclComms_[i] = mcclComms[i];
    work->blockingWait_ = blockingWait_;
    work->opTimeout_ = options_->timeout;
    work->store_ = store_;
  }

  // Future only needs to be created and marked completed with outputs for
  // recv(), but still create future for use cases such as profiling even for
  // send().
  {
    c10::musa::MUSAMultiStreamGuard streamGuard(mcclStreams_[key]);
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Add a callback that runs profiling end callbacks. wrapCallback() in MUSA
  // future blocks the stream this callback runs on the corresponding
  // mcclEndEvents_ ensuring appropriate synchronization.
  if (work->recordFunctionEndCallback_) {
    work->future_->addCallback([work](at::ivalue::Future& /* unused */) {
      work->recordFunctionEndCallback_();
    });
  }

  // Enqueue P2P op so that it can be cancelled by NCCL watchdog
  c10::musa::CaptureStatus capture_status =
      c10::musa::currentStreamCaptureStatusMayInitCtx();

  if (!coalescing_state_ && capture_status == c10::musa::CaptureStatus::None) {
    // Notify graphs before we check the capture status preemptively
    at::musa::MUSAGraph::inc_pending_event_queries();
    workEnqueue(work);
  }

  return work;
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupMCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    OpType opType,
    const char* profilingTitle,
    bool avoidRecordStreams) {
  return collective(
      inputs,
      outputs,
      fn,
      [](std::vector<c10::musa::MUSAStream>&,
         c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {},
      [](std::vector<c10::musa::MUSAStream>&) {},
      opType,
      profilingTitle,
      avoidRecordStreams);
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupMCCL::pointToPoint(
    std::vector<at::Tensor>& tensor,
    Fn fn,
    int peer,
    OpType opType,
    const char* profilingTitle) {
  return pointToPoint(
      tensor,
      fn,
      peer,
      opType,
      [](std::vector<c10::musa::MUSAStream>&,
         c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {},
      [](std::vector<c10::musa::MUSAStream>&) {},
      profilingTitle);
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::allreduce_impl(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  int dev_in_group = 0;
  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream) {
        auto mcclDataType = getMcclDataType(input.scalar_type());
        auto mcclReduceOp = getMcclReduceOp(
            opts.reduceOp, input, mcclDataType, comm, dev_in_group++);
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
      "mccl:all_reduce");
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  check_gpu_tensors_different_devices(tensors);

  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(getBackendName(), std::string()), // process group name
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      0,
      0,
      this->getSize());

  return allreduce_impl(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  auto total_numel = check_gpu_tensors_same_device(tensors);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(getBackendName(), std::string()), // process group name
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce_coalesced", // colName
      total_numel, // inSize
      total_numel, // outSize
      tensors[0].scalar_type(), // dType
      // I'm not sure what in,outSplitSizes mean here.
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      0,
      0,
      this->getSize());

  return allreduce_impl(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  check_gpu_tensors_different_devices(tensors);

  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(getBackendName(), std::string()), // process group name
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "broadcast", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      0,
      0,
      this->getSize());

  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream) {
        const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
        return mcclBcast(
            input.data_ptr(),
            input.numel(),
            getMcclDataType(input.scalar_type()),
            root,
            comm,
            stream.stream());
      },
      OpType::BROADCAST,
      "mccl:broadcast",
      avoidRecordStreams);
}

// _broadcast_oop adds an out-of-place broadcast in PGMCCL
// Custom collectives may be implemented by coalescing broadcast operations
// One use-case is implementing a vector all_gather (all_gather_v)
// where unevenly sized inputs are gathered among participating ranks
// Since all_gather provides an out-of-place API, an all_gather_v
// semantic implemented inside pg_mccl.all_gather also needs to support
// out-of-place, for which an out-of-place broadcast is required to be added
c10::intrusive_ptr<Work> ProcessGroupMCCL::_broadcast_oop(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const BroadcastOptions& opts) {
  check_gpu_tensors_different_devices(outputTensors);
  check_gpu_tensors_different_devices(inputTensors);

  // @lint-ignore CLANGTIDY
  auto tensor = outputTensors.back();
  // @lint-ignore CLANGTIDY
  auto in_tensor = inputTensors.back();
  if (tensor.numel() != in_tensor.numel()) {
    TORCH_CHECK(
        false,
        "Tensor input and output of _broadcast_oop must have the same number of elements ");
  }
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() +
          1), // seq + 1 to match collective increment.
      std::make_tuple(getBackendName(), std::string()), // process group name
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "_broadcast_oop", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      0,
      0,
      this->getSize());

  return collective(
      inputTensors,
      outputTensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream) {
        const auto root = opts.rootRank * inputTensors.size() + opts.rootTensor;
        return mcclBroadcast(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getMcclDataType(input.scalar_type()),
            root,
            comm,
            stream.stream());
      },
      OpType::BROADCAST,
      "mccl:_broadcast_oop");
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  check_gpu_tensors_different_devices(tensors);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(getBackendName(), std::string()), // process group name
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "reduce", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      0,
      0,
      this->getSize());

  int dev_in_group = 0;
  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream) {
        const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
        auto mcclDataType = getMcclDataType(input.scalar_type());
        auto mcclReduceOp = getMcclReduceOp(
            opts.reduceOp, input, mcclDataType, comm, dev_in_group++);
        return mcclReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            mcclDataType,
            mcclReduceOp,
            root,
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
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ReduceOptions& opts) {
  check_gpu_tensors_different_devices(outputTensors);
  check_gpu_tensors_different_devices(inputTensors);
  // @lint-ignore CLANGTIDY
  auto tensor = outputTensors.back();
  // @lint-ignore CLANGTIDY
  auto in_tensor = inputTensors.back();
  if (tensor.numel() != in_tensor.numel()) {
    TORCH_CHECK(
        false,
        "Tensor input and output of _reduce_oop must have the same number of elements ");
  }
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(getBackendName(), std::string()), // process group name
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "_reduce_oop", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      0,
      0,
      this->getSize());

  int dev_in_group{0};
  return collective(
      inputTensors,
      outputTensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream) {
        const auto root = opts.rootRank * inputTensors.size() + opts.rootTensor;
        const auto mcclDataType = getMcclDataType(input.scalar_type());
        const auto mcclReduceOp = getMcclReduceOp(
            opts.reduceOp, input, mcclDataType, comm, dev_in_group++);
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
  check_gpu_tensors_different_devices(inputTensors);
  // @lint-ignore CLANGTIDY
  bool same_size = check_same_size(outputTensors.back());

  if (same_size) {
    auto outputFlattened =
        flatten_for_scatter_gather(outputTensors, inputTensors, size_);
    check_gpu_tensors_different_devices(outputFlattened);

    // @lint-ignore CLANGTIDY
    auto tensor = inputTensors.back();
    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        std::make_tuple(getBackendName(), std::string()), // process group name
        inputTensors, // inputTensors
        outputTensors, // outputTensors
        rank_, // rank
        "all_gather", // colName
        tensor.numel(), // inSize
        tensor.numel() * // outSize
            this->getSize(), // dType
        tensor.scalar_type(),
        std::vector<int64_t>(), // inSplitSizes
        std::vector<int64_t>(), // outSplitSize
        0,
        0,
        this->getSize());

    return collective(
        inputTensors,
        outputFlattened,
        [&](at::Tensor& input,
            at::Tensor& output,
            mcclComm_t comm,
            c10::musa::MUSAStream& stream) {
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
        [](std::vector<c10::musa::MUSAStream>& mcclStreams,
           c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {},
        [&](std::vector<c10::musa::MUSAStream>& mcclStreams) {
          // Copy the flattened output tensors to the outputs.
          for (const auto i : c10::irange(outputTensors.size())) {
            c10::musa::MUSAStreamGuard guard(mcclStreams[i]);
            for (const auto j : c10::irange(outputTensors[0].size())) {
              // See [Sync Streams].
              if (!avoidRecordStreams_) {
                c10::musa::MUSACachingAllocator::recordStream(
                    outputTensors[i][j].storage().data_ptr(), mcclStreams[i]);
              }
              outputTensors[i][j].copy_(outputFlattened[i][j], true);
            }
          }
        },
        OpType::ALLGATHER,
        "mccl:all_gather");
  } else {
    const auto num_devices = outputTensors.size();
    const auto num_reduces = outputTensors[0].size();
    std::vector<c10::intrusive_ptr<Work>> works;
    startCoalescing();
    for (const auto i : c10::irange(num_reduces)) {
      std::vector<at::Tensor> inputs_multi_dev(num_devices);
      std::vector<at::Tensor> outputs_multi_dev(num_devices);
      for (const auto j : c10::irange(num_devices)) {
        // @lint-ignore CLANGTIDY
        outputs_multi_dev[j] = outputTensors[j][i];
        inputs_multi_dev[j] =
            // @lint-ignore CLANGTIDY
            i == (rank_ * num_devices + j) ? inputTensors[j]
                                           : outputs_multi_dev[j];
      }
      auto broadcastOpts = BroadcastOptions{
          static_cast<int64_t>(i / num_devices),
          static_cast<int64_t>(i % num_devices),
          opts.timeout};
      auto work =
          _broadcast_oop(outputs_multi_dev, inputs_multi_dev, broadcastOpts);
      works.push_back(work);
    }
    auto work = endCoalescing();
    return work;
    // return initCoalescedWork(works, rank_, OpType::BROADCAST);
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
  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream) {
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
  check_gpu_tensors_different_devices(outputTensors);
  // @lint-ignore CLANGTIDY
  bool same_size = check_same_size(inputTensors.back());

  if (same_size) {
    // @lint-ignore CLANGTIDY
    auto tensor = outputTensors.back();

    int dev_in_group{0};
    auto inputFlattened =
        flatten_for_scatter_gather(inputTensors, outputTensors, size_);
    check_gpu_tensors_different_devices(inputFlattened);

    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        std::make_tuple(getBackendName(), std::string()), // process group name
        inputTensors, // inputTensors
        outputTensors, // outputTensors
        rank_, // rank
        "reduce_scatter", // colName
        tensor.numel() * this->getSize(), // inSize
        tensor.numel(), // outSize
        tensor.scalar_type(), // dType
        std::vector<int64_t>(), // inSplitSizes
        std::vector<int64_t>(), // outSplitSizes
        0,
        0,
        this->getSize());

    return collective(
        inputFlattened,
        outputTensors,
        [&](at::Tensor& input,
            at::Tensor& output,
            mcclComm_t comm,
            c10::musa::MUSAStream& stream) {
          if (!avoidRecordStreams_) {
            c10::musa::MUSACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          const auto mcclDataType = getMcclDataType(input.scalar_type());
          const auto mcclReduceOp = getMcclReduceOp(
              opts.reduceOp, input, mcclDataType, comm, dev_in_group++);
          return mcclReduceScatter(
              input.data_ptr(),
              output.data_ptr(),
              output.numel(),
              mcclDataType,
              mcclReduceOp,
              comm,
              stream.stream());
        },
        [&](std::vector<c10::musa::MUSAStream>& mcclStreams,
            c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {
          if (avoidRecordStreams_) {
            auto& v = work->stashed_for_allocator_safety_;
            for (const auto i : c10::irange(inputTensors.size())) {
              v->insert(
                  v->end(), inputTensors[i].begin(), inputTensors[i].end());
            }
          }
          // Copy the input tensors to the flattened inputs.
          for (const auto i : c10::irange(inputTensors.size())) {
            c10::musa::MUSAStreamGuard guard(mcclStreams[i]);
            for (const auto j : c10::irange(inputTensors[0].size())) {
              // See [Sync Streams].
              if (!avoidRecordStreams_) {
                c10::musa::MUSACachingAllocator::recordStream(
                    inputTensors[i][j].storage().data_ptr(), mcclStreams[i]);
              }
              inputFlattened[i][j].copy_(inputTensors[i][j], true);
            }
          }
        },
        [&](std::vector<c10::musa::MUSAStream>&) {},
        OpType::REDUCE_SCATTER,
        "mccl:reduce_scatter");
  } else {
    const auto num_devices = inputTensors.size();
    const auto num_reduces = inputTensors[0].size();
    std::vector<c10::intrusive_ptr<Work>> works;
    startCoalescing();
    for (const auto i : c10::irange(num_reduces)) {
      std::vector<at::Tensor> inputs_multi_dev(num_devices);
      std::vector<at::Tensor> outputs_multi_dev(num_devices);
      for (const auto j : c10::irange(num_devices)) {
        // @lint-ignore CLANGTIDY
        inputs_multi_dev[j] = inputTensors[j][i];
        outputs_multi_dev[j] =
            // @lint-ignore CLANGTIDY
            i == (rank_ * num_devices + j) ? outputTensors[j]
                                           : inputs_multi_dev[j];
      }
      auto reduceOpts = ReduceOptions{
          opts.reduceOp,
          static_cast<int64_t>(i / num_devices),
          static_cast<int64_t>(i % num_devices),
          opts.timeout};
      auto work = _reduce_oop(outputs_multi_dev, inputs_multi_dev, reduceOpts);
      works.push_back(work);
    }
    auto work = endCoalescing();
    return work;
    // return initCoalescedWork(works, rank_, OpType::REDUCE);
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
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(getBackendName(), std::string()), // process group name
      inputTensor, // inputTensor
      outputTensor, // outputTensor
      rank_, // rank
      "_reduce_scatter_base", // colName
      tensor.numel() * // inSize
          this->getSize(),
      tensor.numel(), // outSize
      tensor.scalar_type(), // dtype
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      0,
      0,
      this->getSize());

  auto inputs = std::vector<at::Tensor>{inputTensor};
  auto outputs = std::vector<at::Tensor>{outputTensor};

  int dev_in_group = 0;
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
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream) {
        if (!avoidRecordStreams) {
          c10::musa::MUSACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        auto mcclDataType = getMcclDataType(input.scalar_type());
        auto mcclReduceOp = getMcclReduceOp(
            opts.reduceOp, input, mcclDataType, comm, dev_in_group++);
        return mcclReduceScatter(
            input.data_ptr(),
            output.data_ptr(),
            output.numel(),
            mcclDataType,
            mcclReduceOp,
            comm,
            stream.stream());
      },
      [](std::vector<c10::musa::MUSAStream>&,
         c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {},
      [](std::vector<c10::musa::MUSAStream>&) {},
      OpType::_REDUCE_SCATTER_BASE,
      "mccl:_reduce_scatter_base",
      avoidRecordStreams);
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {
  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream) {
        if (!avoidRecordStreams_) {
          c10::musa::MUSACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        auto mcclDataType = getMcclDataType(input.scalar_type());
        auto mcclReduceOp = getMcclReduceOp(
            opts.reduceOp, input, mcclDataType, comm, /*dev_in_group=*/0);
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

c10::intrusive_ptr<Work> ProcessGroupMCCL::barrier(const BarrierOptions& opts) {
  RECORD_PARAM_COMMS(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(getBackendName(), std::string()), // process group name
      rank_, // rank
      "barrier", // colName
      0, // inSize
      0, // outSize
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      0,
      0,
      this->getSize());

  std::vector<at::Device> devices;

  // Use user defined GPU device ids if provided
  if (!opts.device_ids.empty()) {
    for (auto device : opts.device_ids) {
      devices.emplace_back(at::DeviceType::PrivateUse1, device);
    }
  } else if (usedDeviceIdxs_.empty()) {
    // This means there is not yet a MCCL collective being called
    // Here we have to use the best guesses and will use a single GPU to call
    // allreduce to achieve barrier.
    // In case the multiple processes fall into the same node, we use rank to
    // ensure that each process is on a different GPU
    auto numGPUs = c10::musa::device_count();
    int16_t deviceIdx = static_cast<int16_t>(rank_ % numGPUs);
    LOG(INFO) << c10::str(
        "Rank ",
        this->getRank(),
        " using GPU ",
        deviceIdx,
        " to perform barrier as devices used by this process are currently unknown. ",
        "This can potentially cause a hang if this rank to GPU mapping is incorrect.",
        "Specify device_ids in barrier() to force use of a particular device.");
    devices.emplace_back(getDeviceForRank(rank_));
  } else {
    for (auto usedDeviceIdx : usedDeviceIdxs_) {
      devices.emplace_back(at::DeviceType::PrivateUse1, usedDeviceIdx);
    }
  }

  std::vector<at::Tensor> barrierTensors;
  barrierTensors.reserve(devices.size());

  c10::musa::OptionalMUSAGuard gpuGuard;
  for (auto& device : devices) {
    gpuGuard.set_index(device.index());
    barrierTensors.push_back(at::empty(
        {1},
        at::TensorOptions()
            .device(at::DeviceType::PrivateUse1)
            .dtype(at::kByte)));
  }

  // All reduce to achieve the barrier
  auto work = allreduce(barrierTensors);

  // Work will take over barrierTensors
  auto mcclWork = dynamic_cast<ProcessGroupMCCL::WorkMCCL*>(work.get());
  TORCH_CHECK(mcclWork);
  mcclWork->barrierTensors_ = std::move(barrierTensors);

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
  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0) {
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};

    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        std::make_tuple(getBackendName(), std::string()), // process group name
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_all", // colName
        inputTensor.numel(), // inSize
        outputTensor.numel(), // outSize
        inputTensor.scalar_type(), // dType
        std::vector<int64_t>(), // inSplitSizes
        std::vector<int64_t>(), // outSplitSizes
        0,
        0,
        this->getSize());

    return collective(
        inputTensors,
        outputTensors,
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
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};

    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        std::make_tuple(getBackendName(), std::string()), // process group name
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_allv", // colName
        inputTensor.numel(), // inSize
        outputTensor.numel(), // outSize
        inputTensor.scalar_type(), // dType
        inputSplitSizes, // inSplitSizes
        outputSplitSizes, // outSplitSizes
        0,
        0,
        this->getSize());

    return collective(
        inputTensors,
        outputTensors,
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
  auto device = outputTensors[0].device();
  for (const auto r : c10::irange(outputTensors.size())) {
    check_gpu_single_tensor(outputTensors[r]);
    check_gpu_single_tensor(inputTensors[r]);
    TORCH_CHECK(
        device == outputTensors[r].device() &&
            device == inputTensors[r].device(),
        "Tensors must be on the same device")
  }
  std::vector<at::Tensor> inputTensor0 = {inputTensors[0]};
  std::vector<at::Tensor> outputTensor0 = {outputTensors[0]};
  return collective(
      inputTensor0,
      outputTensor0,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          mcclComm_t comm,
          c10::musa::MUSAStream& stream) {
        return all2all_impl(outputTensors, inputTensors, comm, stream);
      },
      [&](std::vector<c10::musa::MUSAStream>&,
          c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work) {
        if (avoidRecordStreams_) {
          // TODO(tyr): remove after test.
          TORCH_CHECK(false, "NOT Supported avoidRecordStreams");
          auto& v = work->stashed_for_allocator_safety_;
          v->insert(v->end(), inputTensors.begin(), inputTensors.end());
          v->insert(v->end(), outputTensors.begin(), outputTensors.end());
        }
      },
      [](std::vector<c10::musa::MUSAStream>&) {},
      OpType::ALLTOALL);
}

// torch_musa P2P comms call mccl Operators directly.
// As we only use the latest mccl version and no need of
// backward compatible.
c10::intrusive_ptr<Work> ProcessGroupMCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /* unused */) {
  check_gpu_tensors_different_devices(tensors);
  auto ret = pointToPoint(
      tensors,
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
      "mccl:send");
  return ret;
}

c10::intrusive_ptr<Work> ProcessGroupMCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /* unused */) {
  check_gpu_tensors_different_devices(tensors);
  auto ret = pointToPoint(
      tensors,
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
      "mccl:recv");
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

// gather impl is assembled by send & recv,
// mainly same as gather in torch/csrc/cuda/mccl.cpp
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
  check_gpu_tensors_different_devices(inputTensors);
  assertSingleElementInput(invalidArgument, inputTensors);

  // @lint-ignore CLANGTIDY
  auto tensor = inputTensors.back();

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

    const auto& options = inputTensors[0].options();
    const auto& sizes = inputTensors[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, outputTensors[0], options, sizes);
    outputs = outputTensors[0];
  } else {
    // if not in the root rank, initialize outputs as empty list
    if (outputTensors.size() != 0) {
      invalidArgument("requires empty output on non-root");
    }
    outputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    outputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(getBackendName(), std::string()), // process group name
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "gather", // colName
      tensor.numel(), // inSize
      tensor.numel() * this->getSize(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      0,
      0,
      this->getSize());

  return collective(
      inputTensors,
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
        return gather_impl(inputTensors[0], outputs, comm, stream, root);
      },
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
  check_gpu_tensors_different_devices(outputTensors);
  assertSingleElementInput(invalidArgument, outputTensors);

  // @lint-ignore CLANGTIDY
  auto tensor = outputTensors.back();

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

    const auto& options = outputTensors[0].options();
    const auto& sizes = outputTensors[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, inputTensors[0], options, sizes);
    inputs = inputTensors[0];
  } else {
    // if not in the root rank, initialize inputTensors as empty place holder
    // with an empty list
    if (inputTensors.size() != 0) {
      invalidArgument("requires empty input on non-root");
    }
    inputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    inputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(getBackendName(), std::string()), // process group name
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "scatter", // colName
      tensor.numel(), // inSize
      tensor.numel() * this->getSize(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      0,
      0,
      this->getSize());

  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      outputTensors,
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
        return scatter_impl(inputs, outputTensors[0], comm, stream, root);
      },
      OpType::SCATTER,
      "mccl:scatter",
      avoidRecordStreams);
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

  // just a wrapper to fit the collective interface
  auto inputs = std::vector<at::Tensor>{input_tensor};
  auto outputs = std::vector<at::Tensor>{output_tensor};

  // for asyncOp = false, we don't want to record streams because we
  // know that the MCCL stream will join back to the "current" stream right
  // after this op. So we might just as well keep the stream ownership of the
  // input/output tensors unchanged. The benefit would be that the
  // allocation/free of the tensors would look deterministic to the "current"
  // stream so that the caching allocator can reuse memory pool for this stream
  // in a clever way.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      inputs,
      outputs,
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

#ifdef USE_MCCL_WITH_UCC
std::shared_ptr<at::DynamicLibrary> ProcessGroupMCCL::uccLib_ = nullptr;
#endif

bool ProcessGroupMCCL::isUCCAvailable() const {
#ifdef USE_MCCL_WITH_UCC
  return (uccPG_ != nullptr);
#else
  return false;
#endif
}

void ProcessGroupMCCL::abortCommsFromMap(
    const std::unordered_map<
        std::string,
        std::vector<std::shared_ptr<MCCLComm>>>& mcclCommsMap,
    std::optional<std::string> abortReason) {
  // The process may control multiple devices, loop through the communicators
  // on each device
  for (auto& it : mcclCommsMap) {
    auto& devName = it.first;
    for (auto& mcclComm : it.second) {
      c10::musa::OptionalMUSAGuard gpuGuard;

      at::DeviceIndex deviceIndex = getIndexFromDeviceKey(devName);
      if (deviceIndex >= 0) {
        gpuGuard.set_index(deviceIndex);
      }

      LOG(INFO) << "ProcessGroupMCCL destroying mcclComm_ on MUSA device: "
                << devName;

      mcclComm->mcclCommAbort(abortReason);

      LOG(INFO) << "ProcessGroupMCCL destroyed  mcclComm on MUSA device: "
                << devName;
    }
  }
}

// Abort all communicators on this rank
bool ProcessGroupMCCL::abort(std::optional<std::string> abortReason) {
  std::lock_guard<std::mutex> lock(mutex_);
  abortCommsFromMap(devMCCLCommMap_, abortReason);
  abortCommsFromMap(mcclIdToCommMap_, abortReason);
  return true;
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
