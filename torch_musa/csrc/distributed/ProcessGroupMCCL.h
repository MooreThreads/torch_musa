#ifndef TORCH_MUSA_CSRC_DISTRIBUTED_PROCESSGROUPMCCL_H_
#define TORCH_MUSA_CSRC_DISTRIBUTED_PROCESSGROUPMCCL_H_

//#ifdef USE_C10D_MCCL

#include <mccl.h>
#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <chrono>
#include <exception>
#include <future>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAEvent.h"
#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAFunctions.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"
#include "torch_musa/csrc/distributed/MCCLUtils.h"

#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#if defined(MARCH_TYPE) && (MARCH_TYPE >= 220)
#define MCCL_BF16_SUPPORTED 1
#else
#define MCCL_BF16_SUPPORTED 0
#endif

#if defined(MARCH_TYPE) && (MARCH_TYPE >= 310)
#define MCCL_FP8_SUPPORTED 1
#else
#define MCCL_FP8_SUPPORTED 0
#endif

namespace c10d {

using DEV_MCCL_COMM_MAP =
    std::unordered_map<std::string, std::vector<std::shared_ptr<MCCLComm>>>;
using UNIQID_MCCL_COMM_MAP = DEV_MCCL_COMM_MAP;

struct DumpPipe {
  DumpPipe(int rank) {
    std::string fileStem =
        getCvarString({"TORCH_MCCL_DEBUG_INFO_PIPE_FILE"}, "");
    if (fileStem.empty() ||
        getCvarInt({"TORCH_MCCL_TRACE_BUFFER_SIZE"}, 0) <= 0) {
      return;
    }
    TORCH_CHECK(!fileStem.empty(), "TORCH_MCCL_DEBUG_INFO_TEMP_FILE is empty");
    std::string filename = c10::str(fileStem, rank, ".pipe");
    TORCH_CHECK(
        unlink(filename.c_str()) != -1 || errno == ENOENT,
        "Error removing existing named pipe ",
        filename);
    TORCH_CHECK(
        mkfifo(filename.c_str(), 0666) != -1,
        "Error creating named pipe ",
        filename);
    fd_ = open(filename.c_str(), O_RDONLY | O_NONBLOCK);
    LOG(INFO) << "Pipe file " << filename
              << " has been opened, write to it to trigger MCCL Debug Dump.";
    TORCH_CHECK(fd_ != -1, "Error opening named pipe ", filename);
  }
  bool shouldDump() {
    if (fd_ == -1) {
      return false;
    }
    char buf[128];
    // non-blocking from O_NONBLOCK above.
    // Ignore EINTR because we already will poll this
    // again later.
    ssize_t bytesRead = read(fd_, &buf, 128);
    return bytesRead > 0;
  }
  ~DumpPipe() {
    if (fd_ != -1) {
      close(fd_);
    }
  }

 private:
  int fd_ = -1;
};

// Environment variable which controls whether we perform a MCCL health check
// which ensures communicators are healthy at the beginning of init.
static std::vector<std::string> ENABLE_MCCL_HEALTH_CHECK = {
    "ENABLE_MCCL_HEALTH_CHECK"};

// Environment variable which controls whether or not wait() is blocking or
// non-blocking.
static std::vector<std::string> TORCH_MCCL_BLOCKING_WAIT = {
    "MCCL_BLOCKING_WAIT"};

// Environment variable which controls whether or not we perform Async Error
// Handling with MCCL.
static std::vector<std::string> TORCH_MCCL_ASYNC_ERROR_HANDLING = {
    "MCCL_ASYNC_ERROR_HANDLING"};

// Environment Variable to control whether Desync Debug is enabled.
// This variable must be set together with MCCL_ASYNC_ERROR_HANDLING.
static std::vector<std::string> TORCH_MCCL_DESYNC_DEBUG = {"MCCL_DESYNC_DEBUG"};

// If set, ProcessGroupMCCL doesn't use recordStream calls to ensure
// caching allocator safety for tensors used on both user-facing and
// internal comm streams.
// This environment variable maybe helpful for avoiding allocator thrashing,
// cause recordStream prevents the caching allocator from repurposing
// allocations passed to collectives until those collectives finish from
// the host's perspective.
static std::vector<std::string> TORCH_MCCL_AVOID_RECORD_STREAMS = {
    "TORCH_MCCL_AVOID_RECORD_STREAMS"};

// Control the interval inside the monitoring thread to check the coordinated
// signal from other ranks, e.g. to dump the debugging information.
static std::vector<std::string> TORCH_MCCL_COORD_CHECK_MILSEC = {
    "TORCH_MCCL_COORD_CHECK_MILSEC"};

// Control the heartbeat monitor of mccl trace for flight record
static std::vector<std::string> TORCH_MCCL_HEARBEAT_MONITOR = {
    "TORCH_MCCL_HEARBEAT_MONITOR"};

// Control how much extra time we will wait for dumping the debugging info
// before we exit and throws timeout exception.
static std::vector<std::string> TORCH_MCCL_WAIT_TIMEOUT_DUMP_MILSEC = {
    "TORCH_MCCL_WAIT_TIMEOUT_DUMP_MILSEC"};

enum ErrorHandlingMode { NoHandling = 0, TearDown = 1, CleanUpOnly = 2 };

constexpr const char* MCCL_BACKEND_NAME = "mccl";

constexpr const char* EXCEPTION_DUMP = "exception_dump";

constexpr const int kWorkStatusUpdatePeriodMs = 30 * 1000; // 30 seconds
class TORCH_API ProcessGroupMCCL : public Backend {
 public:
  class WorkMCCL : public Work, public std::enable_shared_from_this<WorkMCCL> {
   public:
    // Constructor takes a list of MUSA devices
    WorkMCCL(
        const std::vector<at::Device>& devices,
        int rank,
        c10d::OpType opType,
        uint64_t seq,
        const char* profilingTitle = nullptr,
        const c10::optional<std::vector<at::Tensor>>& inputs = c10::nullopt,
        bool desyncDebug = false);
    // Copy constructor doing partial copy without outputs_. Cleanup thread
    // monitors and removes finished works. However it will deadlock when
    // destructs outputs_ tensors who are view tensors in autograd graph.
    WorkMCCL(const WorkMCCL& w);

    ~WorkMCCL() override;

    // Checks if the MCCL kernel has started to execute.
    bool isStarted();

    // Checks if request has completed. In this specific case of MCCL, it checks
    // if the MCCL operation has completed on the GPU in its own MCCL stream.
    // Non-blocking operation.
    bool isCompleted() override;

    bool isSuccess() const override;

    // Same as calling synchronize() for MCCL work.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void abort() override;

    // Let current stream wait on the completing of the MCCL work
    // Throws on exceptions. Blocking operation, which will wait for work
    // completion.
    void synchronize() override;

    // Synchronize streams by blocking each on the MCCL stream
    void synchronizeStreams();

    // Helper function used in MUSA Stream callbacks to complete WorkMCCL
    // objects and throw exceptions when needed.
    void handleMCCLGuard(ErrorHandlingMode asyncErrorHandling);

    // Helper function that checks if the MCCL kernels have finished
    // execution on the GPUs
    bool finishedGPUExecution();

    // Get a Future object that will be marked as completed internally.
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    // Helper function that sets an exception_ptr on the WorkMCCL object.
    void setException(std::exception_ptr exception_ptr);

    // Helper function that returns True if the WorkMCCL object has timed out
    // and False otherwise.
    bool timedOut();

    std::vector<at::Tensor> result() override;

   protected:
    // The cached list of MUSA devices to operate on
    std::vector<at::Device> devices_;

    // The start MUSA events of MCCL operator tracking this work item on
    // multiple MUSA devices. These start MUSA events are needed by desync
    // debugging if enabled.
    std::shared_ptr<std::vector<at::musa::MUSAEvent>> mcclStartEvents_;

    // The end MUSA events of MCCL operator tracking this work item on
    // multiple MUSA devices.
    std::shared_ptr<std::vector<at::musa::MUSAEvent>> mcclEndEvents_;

    // The MCCL communicators used for this work item.
    std::vector<std::shared_ptr<MCCLComm>> mcclComms_;

    // Tensors used for barrier op
    std::vector<at::Tensor> barrierTensors_;

    // Clone of blockingWait_ from ProcessGroupMCCL.
    bool blockingWait_ = false;

    // TODO(tyr) same as ProcessGroupMCCL. (Here is WorkMCCL)
    bool avoidRecordStreams_ = false;

    // Clone of opTimeout_ from ProcessGroupMCCL.
    std::chrono::milliseconds opTimeout_;

    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

    // Record the collective sequential number.
    uint64_t seq_;

    // Indicates if the MCCL start event has been updated to the store trace.
    // This will be used by desync debug.
    bool startTraceUpdated_{false};

    // Wrapper method for the static checkForMCCLErrors which can be overridden
    // for tests.
    virtual std::exception_ptr checkForMCCLErrors(
        const std::vector<std::shared_ptr<MCCLComm>>& MCCLComms) const;

    friend std::ostream& operator<<(
        std::ostream& output,
        const WorkMCCL& workMCCL);

   private:
    // Helper function for synchronize
    void synchronizeInternal(std::chrono::milliseconds timeout);
    // Checks for MCCL errors and sets an appropriate exception_ptr.
    void checkAndSetException();

    // Checks for MCCL errors and throws an appropriate exception.
    void checkAndThrowException();

    // Just checks whether GPU execution has started, without modifying
    // exception_ptr.
    bool startedGPUExecutionInternal() const;

    // Just checks whether GPU execution has completed, without modifying
    // exception_ptr.
    bool finishedGPUExecutionInternal() const;

    // Reference to the store so that we can write aborted communicators
    // to the store.
    c10::intrusive_ptr<Store> store_;

    // Store a reference to MCCL collective's outputs, used by result and to
    // give a more descriptive message when representing the Work as a string.
    std::shared_ptr<std::vector<at::Tensor>> outputs_;

    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;

    // By keeping these refs alive until after the collective's
    // work rejoins the user-facing streams, we achieve caching
    // allocator safety without any recordStream calls.
    // It is important to keep these refs especially in the case of
    // asynchronous communications.
    std::shared_ptr<std::vector<at::Tensor>> stashed_for_allocator_safety_;

    friend class ProcessGroupMCCL;
  };

  class CoalescedWorkMCCL
      : public Work,
        public std::enable_shared_from_this<CoalescedWorkMCCL> {
   public:
    // Constructor takes a list of WorkMCCL works
    CoalescedWorkMCCL(
        std::vector<ProcessGroupMCCL::WorkMCCL> works,
        int rank,
        c10d::OpType opType);

    ~CoalescedWorkMCCL() override;

    // Same as calling synchronize() for MCCL work.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

   protected:
    // The cached list of MUSA devices to operate on
    std::vector<ProcessGroupMCCL::WorkMCCL> works_;

    friend class ProcessGroupMCCL;
  };

  struct Options : Backend::Options {
    // NOTE: timeout in ProcessGroupMCCL::Options denote the timeout for
    // operations. This is only used when blockingWait_ is enabled.
    explicit Options(bool is_high_priority_stream = false);

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false) {
      return c10::make_intrusive<Options>(is_high_priority_stream);
    }

    // Schedule MCCL operations on high priority MUSA streams
    bool is_high_priority_stream;
  };

  // If you wish to create multiple process groups, each with a potentially
  // different rank and size, you can do so by passing a new store instance
  // to each one. If you have only a single store object, you can
  // use the `c10d::PrefixStore` to derive scoped instances.
  // This is also what the Python API in torch.distributed does.
  //
  // The process group instance keeps a reference to the store because
  // it may be used long after the constructor runs. In fact, the constructor
  // doesn't create any MCCL communicators. A single MCCL communicator can
  // only be used on a specific set of devices, and are therefore created
  // on-demand when a collective runs. If another collective is executed later,
  // against a different set of devices, the process group creates another MCCL
  // communicator. These MCCL communicators are cached and reused if possible.
  //
  ProcessGroupMCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());

  // This constructor includes the deprecated `groupName` argument.
  // If you have existing code that uses the `groupName`, you can replace
  // it by specifying a `c10d::PrefixStore(groupName, store)` for store.
  C10_DEPRECATED ProcessGroupMCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      const std::string& groupName,
      c10::intrusive_ptr<Options> options = Options::create())
      : ProcessGroupMCCL(store, rank, size, options) {}

  ~ProcessGroupMCCL() override;

  static c10::intrusive_ptr<Backend> MCCLcreator(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      std::chrono::milliseconds op_time_out);

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  const std::string getBackendName() const override {
    return std::string(MCCL_BACKEND_NAME);
  }

  bool supportsCoalescing() const override {
    return true;
  }

  void startCoalescing() override;

  c10::intrusive_ptr<Work> endCoalescing() override;

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> _broadcast_oop(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const BroadcastOptions& opts = BroadcastOptions());

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> _reduce_oop(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const ReduceOptions& opts = ReduceOptions());

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  static void groupStart();

  static void groupEnd();

  // Unsupported Ops
  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  // Agrees on an initial sequence number for the whole group by having rank 0
  // create it and broadcast it to other ranks using the store.
  void setSequenceNumberForGroup() override;

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  uint64_t getSequenceNumberForGroup() override;

  // Tests if the UCC fallback path is available
  bool isUCCAvailable() const;

  // Helper function for iteratively aborting communicators in the provided map
  void abortCommsFromMap(
      const std::unordered_map<
          std::string,
          std::vector<std::shared_ptr<MCCLComm>>>& mcclCommsMap,
      std::optional<std::string> abortReason);

  bool abort(std::optional<std::string> abortReason = std::nullopt);

 protected:
  // Helper that broadcasts mccl unique ID to all ranks through the store
  void broadcastUniqueMCCLID(
      mcclUniqueId* mcclID,
      bool isSingleP2POp,
      const std::string& devicesKey,
      int p2pRank);

  // Helper that either looks up the cached MCCL communicators or creates
  // a new set of MCCL communicators as a cache entry
  std::vector<std::shared_ptr<MCCLComm>>& getMCCLComm(
      const std::string& devicesKey,
      const std::vector<at::Device>& devices,
      OpType opType,
      int p2pRank = 0,
      bool isSendRecvSelf = false);

  // Wrapper method which can be overridden for tests.
  virtual std::exception_ptr checkForMCCLErrors(
      const std::vector<std::shared_ptr<MCCLComm>>& mcclComms);

  virtual c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL> initWork(
      std::vector<at::Device> devices,
      int rank,
      OpType opType,
      const char* profilingTitle = nullptr,
      const c10::optional<std::vector<at::Tensor>>& inputs = c10::nullopt);

  virtual c10::intrusive_ptr<ProcessGroupMCCL::CoalescedWorkMCCL>
  initCoalescedWork(
      const std::vector<c10::intrusive_ptr<Work>>& works,
      int rank,
      OpType opType);

 private:
  // Helper that encapsulates work shared across all collective communication
  // primitives.  The callbacks have the following signatures:
  //
  //    mcclResult_t fn(at::Tensor& input, at::Tensor& output,
  //                    mcclComm_t, c10::musa::MUSAStream&);
  //    void {pre,post}(std::vector<c10::musa::MUSAStream&>);
  template <typename Fn>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      OpType opType,
      const char* profilingTitle = nullptr,
      bool avoidRecordStreams = false);
  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType,
      const char* profilingTitle = nullptr,
      bool avoidRecordStreams = false);

  // Helper that encapsulates work shared across point-to-point communication
  // primitives. It is the same structure as the helper used for collective
  // communicaiton primitives.
  template <typename Fn>
  c10::intrusive_ptr<Work> pointToPoint(
      std::vector<at::Tensor>& tensor,
      Fn fn,
      int peer,
      OpType opType,
      const char* profilingTitle = nullptr);
  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> pointToPoint(
      std::vector<at::Tensor>& tensor,
      Fn fn,
      int peer,
      OpType opType,
      PreProcess pre,
      PostProcess post,
      const char* profilingTitle);

  c10::intrusive_ptr<Work> allreduce_impl(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions());

  // Checks for MCCL errors on each of the communicators and returns an
  // appropriate exception_ptr (nullptr if no errors).
  static std::exception_ptr checkForMCCLErrorsInternal(
      const std::vector<std::shared_ptr<MCCLComm>>& mcclComms);

  // Function that runs as part of a separate thread and checks for errors on
  // MCCL communicators. We need a separate thread to check for MCCL errors
  // since we can't rely on the user calling certain methods like wait(),
  // isCompleted() etc. to detect and remediate errors. In addition to this, we
  // need a mechanism to safely abort and remove MCCL communicators from our
  // cache. This can be done cleanly by having a thread for the ProcessGroupMCCL
  // class. Attempting to modify the communicator cache from the WorkMCCL class
  // might run into issues with object lifetime since the ProcessGroupMCCL
  // object might get destroyed before the WorkMCCL object.
  void mcclCommWatchdog();

  void mcclCommWatchdogInternal();

  // This function iterates through the list of WorkMCCL objects in the
  // workList_ corresponding to incomplete collectives and then aborts MCCL
  // communicators associated with timed out collectives.
  void abortTimedOutCollectives(
      std::unordered_set<std::string>& abortedCommIds);

  // Performs a health check by initializing dummy MCCL communicators and then
  // destroying them. This will help indicate and signal any MCCL-related issues
  // prior to the first collective. The actual initialization and subsequent
  // destruction is ran on a separate thread and the main thread is signalled
  // about timeouts/errors to report to the application.
  void runHealthCheck();

  // Destroys initialized MCCL communicators in devMCCLComMap_ given by input
  // key. Throws if there are no communicators to destroy. Also removes
  // communicators from the cache and clears used device indices.
  void destroyMCCLComms(const std::string& devMCCLCommMapKey);

  void workCleanupLoop();

  void heartbeatMonitor();

  bool dumpDebuggingInfo();

  std::string dumpMCCLTrace(
      bool includeCollectives,
      bool includeStackTraces,
      bool onlyActive);

  std::string getMCCLWatchdogTimeoutErrorMsg(const std::string& extraMsg);

  void waitForFutureOrTimeout(
      std::future<bool>& fut,
      const std::chrono::milliseconds& timeOutMilSec,
      const std::string& futDescription,
      bool throwException,
      bool log);

 protected:
  static const int64_t kWatchdogThreadSleepMillis;
  static const int64_t kWorkCleanupThreadSleepMillis;
  static const int64_t kHeartBeatThreadSleepMillis;

  // The store is used to broadcast the MCCL unique ID of rank 0.
  c10::intrusive_ptr<Store> store_;

  bool storeError_{false};

  const c10::intrusive_ptr<Options> options_;

  // The number of MCCL communicators that have been created during
  // the lifetime of this process group. This sequence number is
  // used to scope keys used in the store.
  uint64_t mcclCommCounter_{0};

  // The store keys to trace the last MCCL collective kernel MUSA events - start
  // event and end event respectively. These are used to do desync root cause
  // analysis.
  const std::string traceKeyStart_;
  const std::string traceKeyEnd_;

  // The MCCL communicator that the process group has cached.
  //
  // For collective operations:
  // The key is a list of GPU devices that an operation is operating on
  // The GPU devices are stored in a device sequence and the cache MCCL
  // communicator is associated with this GPU device sequence
  //
  // e.g. If the process group op only uses device 0, then the value of
  // the used device string stored (value of the hashmap) would be "0".
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 1, 2, 3, 4, 5, 6, 7 separately,
  //      then the value of the used device string (key) stored would be
  //      "0,1,2,3,4,5,6,7"
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 4, 5, 6, 7, 1, 2, 3 separately,
  //      then the value of the used device string stored would be
  //      "0,4,5,6,7,1,2,3"
  //
  //      Note that the order of the device for the tensor list matters.
  //
  // For point-to-point operations:
  // The key is a string of my current rank and the peer process rank.
  // e.g. If process 1 and process 2 are involved in a point-to-point
  // communication, the key will be "1:2" on both processes. Note: this is for
  // the scenario where there is only 1 GPU per process. When it comes to
  // multiple GPUs per process, this part may need to redesigned.
  DEV_MCCL_COMM_MAP devMCCLCommMap_;

  // Map from mcclUniqueId to appropriate communicator.
  UNIQID_MCCL_COMM_MAP mcclIdToCommMap_;

  // Mutex to guard maps like devMCCLCommMap_ and mcclIdToCommMap_.
  std::mutex mutex_;

  // Watchdog thread which looks for errors on the cached MCCL communicators.
  std::thread mcclCommWatchdogThread_;

  // Monitor thread which checks the heartbeat of Watchdog thread.
  // If the monitor thread finds there is no heartbeat, it will dump debug info
  // and then kill the watchdog thread to avoid hang.
  std::thread mcclHeartbeatMonitorThread_;

  // Whether or not we should terminate the watchdog and workCleanup threads.
  std::atomic<bool> terminateProcessGroup_;

  // Condition variable to control how long the watchdog thread waits.
  std::condition_variable watchdogCV_;

  // Mutex for watchdog.
  std::mutex watchdogCVMutex_;

  // Thread that removes MCCL Work upon timeout
  std::thread workCleanupThread_;

  // Mutex to Guard workMetaList_
  std::mutex workMetaListMutex_;

  // Condition Variable for timeout thread sleep
  std::condition_variable workMetaListCV_;

  // Vector to Store WorkMCCL pointers
  std::list<ProcessGroupMCCL::WorkMCCL> workMetaList_;

  // Add Work Pointer to workVector
  void workEnqueue(c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>);

  // The MUSA steams used by MCCL kernels
  std::unordered_map<std::string, std::vector<c10::musa::MUSAStream>>
      mcclStreams_;

  // The MUSA events used to sync MCCL streams
  std::unordered_map<std::string, std::vector<at::musa::MUSAEvent>> mcclEvents_;

  // Device Indexes used for all collectives in this group
  std::set<int> usedDeviceIdxs_;

  // Flag to denote if a coalescing groupStart/groupEnd block is active
  int coalescing_state_ = 0;

  // Stores device indexes for all collectives run inside a coalescing block
  std::vector<std::vector<at::Device>> coalescedDevices_;

  std::vector<std::vector<std::shared_ptr<MCCLComm>>> coalescedComms_;

  // map from the key: "group name + pg counter (ID)" to the
  // unique MCCL ID count. This needs to be group and pg specific
  //
  // For each process group, we need a uniform unique MCCL ID counter to ensure
  // that MCCL operation in this process group can be completed successfully.
  // Since each process group ID belongs to a group name, the key to this map
  // is a combination of group name and ProcessGroupMCCL ID.
  static std::unordered_map<std::string, ssize_t> pgUniqueMCCLIDCnt_;

  // map from group name to the pg counter (ID) within that group
  //
  // For each group with the "group name" (which is the key), we need to
  // keep track of a unique process group ID when creating a new
  // ProcessGroupMCCL for this "group name". Therefore, the value of this
  // map keeps the unique ProcessGroupMCCL's ID for a specific group with
  // the "group name". The reason we need a per-group process group ID counter
  // is that different group can have different ranks and we need ensure that
  // each group has its own uniform process group ID for all its ranks.
  static std::unordered_map<std::string, ssize_t> processGroupCounterMap_;

  // Whether or not wait() and synchronize() are blocking operations that wait
  // for the operation to complete.
  bool blockingWait_ = false;

  // Whether or not the workCleanupThread is used to perform async error
  // handling.
  ErrorHandlingMode asyncErrorHandling_ = NoHandling;

  // Whether or not to enable timeout root cause analysis.
  bool desyncDebug_;

  // Whether or not TORCH_MCCL_AVOID_RECORD_STREAMS was set
  bool avoidRecordStreams_ = false;

  // Set of communicators that this process group has aborted and their
  // mcclUniqueId has been written to the store. We don't need a lock
  // for this map since only the watchdog thread accesses this set. The
  // set contains the string representation of mcclUniqueId.
  std::unordered_set<std::string> abortedComms_;

  // The number of active mcclGroupStart() calls. This counter will be increased
  // by 1 when mcclGroupStart() is called and decreased by 1 when mcclGroupEnd()
  // is called.
  static thread_local uint64_t mcclActiveGroupCounter_;

  // Counting for the sequential number of MCCL collective call.
  uint64_t seq_{0};

  // Whether or not to dump debug info on exception including both watchdog
  // timeout and mccl errors.
  bool dumpOnTimeoutOrEx_;

  // The number of ProcessGroupMCCL created on the current rank.
  size_t local_id_{0};

  // Whether or not we should terminate the heartbeat monitoring threads.
  std::atomic<bool> terminateHeartbeatMonitorThread_;

  // Condition Variable for monitor thread to wake up early
  std::condition_variable monitorWakeUpCV_;

  // Mutex to Guard monitorWakeUpCV_
  std::mutex monitorMutex_;

  // Size of ring buffer where we store MCCL Traces for debugging.
  int mcclTraceBufferSize_;

  // The time interval used for deciding whether there is no watchdog heartbeat.
  int heartbeatTimeoutInSec_{60 * 8}; // 8 min

  // Interval of check coordinated signals in ProcessGroupMCCL from other ranks
  // e.g., trigger the dump of the debugging info for timeout when notified.
  int coordCheckIntervalMilSec_;

  // timeout for the dump to finish.
  int waitTimeoutDumpInMilSec_;

  // the flag of heartbeat monitor for mccl status
  bool mcclHeartBeatMonitor_{false};

  // Heartbeat of watchdog thread.
  std::atomic_uint64_t heartbeat_;

  // This is the signal from watchdog threads to indicate whether the monitor
  // thread should dump. Making it static so that it is accessiable from all the
  // PGs. With this flag, monitor thread would dump debug info under any one of
  // the three conditions:
  //
  // 1: watchdog thread of any PG detects a collective timeout.
  // 2: timeout signal is received from other ranks through tcpstore.
  // 3: current PG's watchdog heartbeat timeout occurs.
  //
  // Note that only the monitor thread from PG0 will dump the debug info for
  // case one and two so that the debug info is only dumped once.
  static std::atomic<bool> shouldDump_;

  std::chrono::time_point<std::chrono::steady_clock> lastWorkListUpdateTime_;

  std::shared_ptr<ProcessGroupStatus> pgStatus_ =
      std::make_shared<ProcessGroupStatus>();

}; // class ProcessGroupMCCL

} // namespace c10d

void registerProcessGroupMCCL(PyObject* mod);
//#endif // USE_C10D_MCCL

#endif // TORCH_MUSA_CSRC_DISTRIBUTED_PROCESSGROUPMCCL_H_
