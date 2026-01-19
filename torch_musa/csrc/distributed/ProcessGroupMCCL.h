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
#include <torch/csrc/distributed/c10d/logger.hpp>

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

// Control whether to always use high priority streams
static std::vector<std::string> TORCH_MCCL_HIGH_PRIORITY = {
    "TORCH_MCCL_HIGH_PRIORITY"};

// Control whether or not wait() is blocking or non-blocking.
static std::vector<std::string> TORCH_MCCL_BLOCKING_WAIT = {
    "TORCH_MCCL_BLOCKING_WAIT",
    "MCCL_BLOCKING_WAIT"};

// TODO: We want to eventually remove this variable and make users to use
// the default value (3 - SkipCleanUp).
// Control whether or not we perform Async Error Handling with MCCL.
static std::vector<std::string> TORCH_MCCL_ASYNC_ERROR_HANDLING = {
    "TORCH_MCCL_ASYNC_ERROR_HANDLING",
    "MCCL_ASYNC_ERROR_HANDLING"};

// Control whether dumping debug info on watchdog
// timeout is enabled. This variable must be set together with
// TORCH_MCCL_ENABLE_MONITORING=1 and TORCH_MCCL_TRACE_BUFFER_SIZE > 0.
static std::vector<std::string> TORCH_MCCL_DUMP_ON_TIMEOUT = {
    "TORCH_MCCL_DUMP_ON_TIMEOUT"};

// Control whether to propagate MCCL errors to all ranks through TCPStore.
static std::vector<std::string> TORCH_MCCL_PROPAGATE_ERROR = {
    "TORCH_MCCL_PROPAGATE_ERROR"};

// Control whether Desync Debug is enabled. This variable must be set
// together with TORCH_MCCL_ASYNC_ERROR_HANDLING.
static std::vector<std::string> TORCH_MCCL_DESYNC_DEBUG = {
    "TORCH_MCCL_DESYNC_DEBUG",
    "MCCL_DESYNC_DEBUG"};

// Enable recording start-events for all ProcessGroupMCCL collectives, and
// compute accurate collective timing per-collective. (Note: end-events are
// recorded by default. Turn on this flag can increase chances of a watchdog
// hang due to performing a MUSA event query which eventually calls
// musaEventElapsedTime() API.
static std::vector<std::string> TORCH_MCCL_ENABLE_TIMING = {
    "TORCH_MCCL_ENABLE_TIMING",
    "MCCL_ENABLE_TIMING"};

// Enable monitoring thread which aborts the process when the ProcessGroupMCCL
// Watchdog thread gets stuck and no heartbeat is detected after
// TORCH_MCCL_HEARTBEAT_TIMEOUT_SEC. This can happen due to calling MUSA/MCCL
// APIs that may hang. It is Useful to prevent jobs being stuck for a prolonged
// time than necessary tying up cluster resources.
static std::vector<std::string> TORCH_MCCL_ENABLE_MONITORING = {
    "TORCH_MCCL_ENABLE_MONITORING"};

// Control the watchdog heartbeat timeout period after which the monitoring
// thread will abort the process.
static std::vector<std::string> TORCH_MCCL_HEARTBEAT_TIMEOUT_SEC = {
    "TORCH_MCCL_HEARTBEAT_TIMEOUT_SEC"};

// Whether to rethrow MUSA Errors in the watchdog (default true)
static std::vector<std::string> TORCH_MCCL_RETHROW_MUSA_ERRORS = {
    "TORCH_MCCL_RETHROW_MUSA_ERRORS"};

// The maximum number of events we store in the flight recorder's ring buffer.
// (One event could be the start or end of a collective, for example).
static std::vector<std::string> TORCH_MCCL_TRACE_BUFFER_SIZE = {
    "TORCH_MCCL_TRACE_BUFFER_SIZE"};

// If set, ProcessGroupMCCL doesn't use recordStream calls to ensure
// caching allocator safety for tensors used on both user-facing and
// internal comm streams.
// This environment variable maybe helpful for avoiding allocator thrashing,
// cause recordStream prevents the caching allocator from repurposing
// allocations passed to collectives until those collectives finish from
// the host's perspective.
static std::vector<std::string> TORCH_MCCL_AVOID_RECORD_STREAMS = {
    "TORCH_MCCL_AVOID_RECORD_STREAMS"};

// Control how much extra time we will wait for dumping the debugging info
// before we exit and throws timeout exception.
static std::vector<std::string> TORCH_MCCL_WAIT_TIMEOUT_DUMP_MILSEC = {
    "TORCH_MCCL_WAIT_TIMEOUT_DUMP_MILSEC"};

// Control the interval inside the monitoring thread to check the coordinated
// signal from other ranks, e.g. to dump the debugging information.
static std::vector<std::string> TORCH_MCCL_COORD_CHECK_MILSEC = {
    "TORCH_MCCL_COORD_CHECK_MILSEC"};

// Whether to log C++ stack traces on unclean shutdown (default true)
static std::vector<std::string> TORCH_MCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN = {
    "TORCH_MCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN"};

// Control whether to use CudaEventCache for the collective in watchdog thread.
// We noticed in the past when musa global lock is held, destroying CudaEvent
// can cause a hang.
static std::vector<std::string> TORCH_MCCL_MUSA_EVENT_CACHE = {
    "TORCH_MCCL_MUSA_EVENT_CACHE"};

// MCCL does not support ScalableInit
// Control the number of ranks each root can cover during MCCL comm init.
static std::vector<std::string> TORCH_MCCL_RANKS_PER_ROOT = {
    "TORCH_MCCL_RANKS_PER_ROOT"};

// TODO(chen.feng): enable nanCheck
static std::vector<std::string> TORCH_MCCL_NAN_CHECK = {"TORCH_MCCL_NAN_CHECK"};

constexpr const char* MCCL_BACKEND_NAME = "mccl";

constexpr const char* kStoreDumpKey = "exception_dump";

constexpr const char* kStoreErrorSignalKey = "remote_error";

constexpr const int kWorkStatusUpdatePeriodMs = 30 * 1000; // 30 seconds

constexpr auto kProcessGroupMCCLDefaultTimeout =
    std::chrono::milliseconds(10 * 60 * 1000);

// NoHandling: do not handle asynchronous MCCL errors
// TearDown: tear down process upon error, see `WorkMCCL::handleException`
// CleanUpOnly: just clean up collectives and abort communicators without
// tearing down process SkipCleanUp: (this is a temporary option and can be
// removed in future) tear down process without cleaning up MCCL communicators.
// This should be used as a last resort in case `mcclCommAbort` itself is
// hanging
enum ErrorHandlingMode {
  NoHandling = 0,
  TearDown = 1,
  CleanUpOnly = 2,
  SkipCleanUp = 3
};

#define SHOULD_CLEAN_UP(a) (a != NoHandling && a != SkipCleanUp)

#define SHOULD_TEAR_DOWN(a) (a != NoHandling && a != CleanUpOnly)

class TORCH_API ProcessGroupMCCL : public Backend {
 public:
  class WorkMCCL : public Work, public std::enable_shared_from_this<WorkMCCL> {
   public:
    // Constructor takes a list of MUSA devices
    WorkMCCL(
        std::string pgUID,
        std::string pgDesc,
        at::Device& device,
        int rank,
        OpType opType,
        uint64_t seq,
        bool isP2P = false,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputs = std::nullopt,
        bool desyncDebug = false,
        bool enableTiming = false,
        bool musaEventCacheEnabled = false,
        DebugLevel distDebugLevel = DebugLevel::Off);
    // Copy constructor doing partial copy without outputs_. Cleanup thread
    // monitors and removes finished works. However it will deadlock when
    // destructs outputs_ tensors who are view tensors in autograd graph.
    WorkMCCL(const WorkMCCL& w);

    ~WorkMCCL() override = default;

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
    void synchronizeStream();

    // Helper function to handle exception (throw if needed).
    void handleException(ErrorHandlingMode asyncErrorHandling);

    // Helper function that checks if the MCCL kernels have finished
    // execution on the GPUs
    bool finishedGPUExecution();

    // Get a Future object that will be marked as completed internally.
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    // Get a Future result of each work (e.g. success, different error types).
    // instead of the tensor output.
    c10::intrusive_ptr<c10::ivalue::Future> getFutureResult() override;

    float getDuration() const override;

    uint64_t getSequencenumber() const override;

    const std::string& logPrefix() const;

    // Helper function that sets an exception_ptr on the WorkMCCL object.
    void setException(std::exception_ptr exception_ptr);

    // Helper function that returns True if the WorkMCCL object has timed out
    // and False otherwise.
    // In case of timeout, set exception on the WorkMCCL object.
    bool checkTimeout(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt);

    // Print the traceback of the collective at call time
    void printTraceback() const;

    std::vector<at::Tensor> result() override;

   protected:
    // The process group unique id
    std::string pgUID_;

    // The process group description
    std::string pgDesc_;

    // The cached list of MUSA devices to operate on
    at::Device device_;

    // The start MUSA event of MCCL operator tracking this work item. These
    // start MUSA events are needed by desync debugging if enabled.
    std::shared_ptr<at::musa::MUSAEvent> mcclStartEvent_;

    // The end MUSA event of MCCL operator tracking this work item.
    std::shared_ptr<at::musa::MUSAEvent> mcclEndEvent_;

    // The MCCL communicator used for this work item.
    std::shared_ptr<MCCLComm> mcclComm_;

    // whether this work is a barrier op
    bool isBarrierOp_{false};

    // Clone of blockingWait_ from ProcessGroupMCCL.
    bool blockingWait_{false};

    // Clone of avoidRecordStreams_ from ProcessGroupMCCL.
    bool avoidRecordStreams_{false};

    // Clone of opTimeout_ from ProcessGroupMCCL.
    std::chrono::milliseconds opTimeout_{};

    // Ephemeral timeouts are owned by exactly one work,
    // and reset after that work completes.
    // There may be more than one ephemeral timeout active at the same time,
    // and this variable is used to track the ownership of ephemeral timeout.
    std::chrono::milliseconds ownedEphermeralTimeout_ =
        std::chrono::milliseconds(0);

    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

    // Record the collective sequential number.
    uint64_t seq_;
    bool isP2P_;

    // Indicates if the MCCL start event has been updated to the store trace.
    // This will be used by desync debug.
    bool startTraceUpdated_{false};

    // Record collective sizes for debug. We only record the size on the first
    // device as multi-device per process is deprecated
    size_t numelIn_ = -1;
    size_t numelOut_ = -1;

    // Wrapper method for the static checkForMCCLErrors which can be overridden
    // for tests.
    virtual std::exception_ptr checkForMCCLErrors();

    friend std::ostream& operator<<(
        std::ostream& output,
        const WorkMCCL& workMCCL);

   private:
    // Checks for MCCL errors and sets an appropriate exception_ptr.
    void checkAndSetException();

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

    // TORCH_MCCL_AVOID_RECORD_STREAMS implementation helper.
    // Stores references to participating non-output tensors (ie inputs,
    // flattened intermediates).
    // We'll clear this list in synchronizeStream, just after user-facing
    // stream(s) are synced with the mccl work stream(s).
    // By keeping these refs (as well as outputs_) alive until after the
    // collective's work rejoins the user-facing streams, we achieve
    // caching allocator safety without any recordStream calls.
    // For in-place collectives, some refs stashed here may alias outputs_,
    // but that doesn't do any harm.
    std::shared_ptr<std::vector<at::Tensor>> stashed_for_allocator_safety_;

    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;

    // the future result (e.g., success or failure) of the work
    c10::intrusive_ptr<at::ivalue::Future> futureWorkResult_;

    bool timingEnabled_;
    // unique id used to tell the trace buffer that this
    // work has completed
    std::optional<uint64_t> trace_id_;
    DebugLevel distDebugLevel_;

    friend class ProcessGroupMCCL;
  };

  class MUSAEventCache
      : public std::enable_shared_from_this<ProcessGroupMCCL::MUSAEventCache> {
   public:
    MUSAEventCache();
    std::shared_ptr<at::musa::MUSAEvent> create(bool timing);
    static std::shared_ptr<ProcessGroupMCCL::MUSAEventCache> get(
        at::DeviceIndex device);

   private:
    std::mutex cacheMutex_;
    // NOTE: We intentionally store raw pointers so that
    // we do not attempt to destroy the event objects on process exit,
    // because musa may be gone.
    std::array<std::deque<at::musa::MUSAEvent*>, 2>
        eventsArray_; // 0 for timing=false, 1 for timing=true
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

    // Optional "parent" backend and color to create communicators from
    // via `ncclCommSplit`
    std::shared_ptr<ProcessGroupMCCL> split_from;
    // Color to use for `ncclCommSplit`, values:
    // * Non-negative value: in group;
    // * NCCL_SPLIT_NOCOLOR (-1): not in group;
    // * NCCL_SPLIT_NOCOLOR - 1: uninitialized.
    // [Note 1]: the type must be `int` instead of `int64_t` because MCCL API
    // accepts int. Otherwise, an implicit conversion may happen at the API call
    // and the value may become negative.
    // [Note 2]: this member is pybinded to Python, the value passed from Python
    // must be within the numerical range of C++ int. Otherwise, Python will
    // raise a RuntimeError saying type is incompatible. See also
    // `_process_group_color` in `distributed_c10d.py`.

    int split_color{-2};
    std::vector<uint64_t> global_ranks_in_group;
    std::string group_name;
  };

  // Helper class related to TORCH_MCCL_DESYNC_DEBUG
  class DesyncDebugger {
   public:
    // Initialize and enable DesyncDebugger
    void init(int rank, int size, c10::intrusive_ptr<Store> store);

    // Run desync debug. This function is called by watchdog at time of timeout.
    void run();

    // Log work start to store.
    void logWorkStart(WorkMCCL& work);

    // Log work end to store.
    void logWorkEnd(WorkMCCL& work);

   private:
    // Whether desync debug is enabled.
    // If false, all functions are no-op.
    bool enabled_{false};

    // From ProcessGroupMCCL
    int rank_;
    int size_;

    // Reference to the store so that we can log start/end event.
    c10::intrusive_ptr<Store> store_;

    // The store keys to trace the last MCCL collective kernel MUSA events -
    // start event and end event respectively. These are used to do desync root
    // cause analysis.
    std::string traceKeyStart_;
    std::string traceKeyEnd_;
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
      c10::intrusive_ptr<Store> store,
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
      : ProcessGroupMCCL(store, rank, size, std::move(options)) {}

  ~ProcessGroupMCCL() override;

  // This function returns a local uid for ProcessGroupMCCL.
  uint64_t getUid() {
    return static_cast<uint64_t>(local_id_);
  }

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

  bool supportsSplitting() const override {
    return false;
  }

  bool supportsCoalescing() const override {
    return true;
  }

  void startCoalescing() override;

  c10::intrusive_ptr<Work> endCoalescing() override;

  // For specifying a composite optype, such as ALLGATHER and REDUCE_SCATTER
  c10::intrusive_ptr<Work> endCoalescing(OpType optype);

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> _broadcast_oop(
      at::Tensor& outputTensors,
      at::Tensor& inputTensors,
      const BroadcastOptions& opts = BroadcastOptions());

  //   c10::intrusive_ptr<Work> allreduce_sparse(
  //       std::vector<at::Tensor>& tensors,
  //       const AllreduceOptions& opts = AllreduceOptions()) override;

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
      at::Tensor& outputTensors,
      at::Tensor& inputTensors,
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

  void groupEndNonblocking(const std::shared_ptr<MCCLComm>& comm);

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

  // Return the total number of splits the communicators held by this process
  // group have performed.  Counts ncclCommCreateFromRanks() for ncclx v2.21.5+
  uint64_t getCommSplitCounter() const;

  void enableCollectivesTiming() override;

  // Helper function for iteratively aborting communicators in the provided map
  void abortCommsFromMap(
      std::unordered_map<std::string, std::shared_ptr<MCCLComm>>& mcclCommsMap,
      const std::optional<std::string>& abortReason);

  int initIntraNodeComm();

  // Destroy (shutdown) this backend -- normal exit.
  void shutdown() override;

  // Provides an API to abort the ProcessGroup (similar to mcclCommAbort)
  // instead of relying on ProcessGroupMCCL destructor.
  void abort() override;

  void eagerConnectSingleDevice(at::Device device) override;

  void performNocolorSplit(at::Device device);

  // If all comms on this PG are fully initialized, return true.
  bool isInitialized();

  ErrorType getError() override;

  //   std::shared_ptr<c10::Allocator> getMemAllocator() override;

  //   // Allocate tensor from communication-optimized memory pool
  //   at::Tensor allocateTensor(long size, at::TensorOptions options = {})
  //   override;

  //   // Whether tensor allocation from MCCL memory pool is supported
  //   bool supportsTensorAlloc(c10::DeviceIndex deviceIdx) override;

  //   // Performs MCCL user buffer registration for all buffers in
  //   // the given MemPool
  //   void registerMemPool(c10::musa::MemPool* pool);

  //   // Performs MCCL user buffer de-registration for all buffers in
  //   // the given MemPool
  //   void deregisterMemPool(c10::musa::MemPool* pool);

  // This method adds a temporary extension for the timeout period,
  // applying to all collectives between the calling of this API and
  // the completion of the first collective on the GPU. While this feature
  // provides flexibility in specific scenarios, it introduces statefulness
  // to timeout setting. Therefore, it is advisable to use this API sparingly
  // and consider alternative approaches, such as directly setting the timeout
  // or utilizing a barrier collective (one can set any timeout to the barrier),
  // whenever feasible.
  void addEphemeralTimeout(const std::chrono::milliseconds& timeout);

  // This function is only intended for testing purposes because we don't
  // want to expose the `WorkMCCL` via pybind. It verifies whether the
  // `opTimeout_` of the provided WorkMCCL instance is the same as the specified
  // timeout.
  bool verifyWorkTimeoutForTest(
      const c10::intrusive_ptr<Work>& work,
      const std::chrono::milliseconds& timeout);

 protected:
  // Helper that broadcasts mccl unique ID to all ranks through the store
  void broadcastUniqueMCCLID(
      mcclUniqueId* mcclID,
      bool isSingleP2POp,
      const std::string& devicesKey,
      int p2pRank);

  // Helper that allgathers mccl unique IDs to all ranks through the store
  void allgatherUniqueMCCLIDs(
      int rootIdx,
      mcclUniqueId* mcclID,
      std::vector<mcclUniqueId>& mcclIDs);

  // Helper that looks up the cached MCCL communicators only
  std::shared_ptr<MCCLComm> getMCCLComm(const std::string& deviceKey);

  std::shared_ptr<MCCLComm> initMCCLComm(
      const std::string& deviceKey,
      at::Device& device,
      OpType opType,
      int p2pRank = 0,
      bool isSendRecvSelf = false);

  // Wrapper method which can be overridden for tests.
  virtual std::exception_ptr checkForMCCLErrors(
      std::shared_ptr<MCCLComm>& mcclComm);

  // Ensure thaht if record is True, the work obj will be enqueued via
  // workEnqueue
  virtual c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL> initWork(
      at::Device& device,
      int rank,
      OpType opType,
      bool isP2P,
      const char* profilingTitle = nullptr,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {},
      bool record = false);

  // In the timeout case and we will dump debug info such as the MCCL flight
  // recorder to storage. Down the road, if we have more complicated or blocking
  // operations, we might need to use a side thread to do it.
  bool dumpDebuggingInfo(bool includeStackTrace = true);

  // Abort all communicators on this rank.
  bool abortComms(const std::optional<std::string>& abortReason = std::nullopt);

  // A helper function to check if nonblocking API mode should be used.
  // Use this helper instead of directly checking `useNonblocking_` variable.
  bool useNonblocking();

 private:
  int globalRankStart;
  int globalRankStride;
  // Helper that encapsulates work shared across all collective communication
  // primitives.  The callbacks have the following signatures:
  //
  //    mcclResult_t fn(at::Tensor& input, at::Tensor& output,
  //                    mcclComm_t, c10::musa::MUSAStream&);
  //    void {pre,post}(std::vector<c10::musa::MUSAStream&>);
  template <typename Fn>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      OpType opType,
      const char* profilingTitle = nullptr,
      bool avoidRecordStreams = false,
      bool nanCheck = true);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType,
      const char* profilingTitle = nullptr,
      bool avoidRecordStreams = false,
      bool nanCheck = true);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType,
      const char* profilingTitle = nullptr,
      bool avoidRecordStreams = false,
      bool nanCheck = true);

  template <typename Fn>
  c10::intrusive_ptr<Work> collectiveCoalesced(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      OpType opType,
      const char* profilingTitle = nullptr,
      bool avoidRecordStreams = false);

  // Helper that encapsulates work shared across point-to-point communication
  // primitives. It is the same structure as the helper used for collective
  // communicaiton primitives.
  template <typename Fn>
  c10::intrusive_ptr<Work> pointToPoint(
      at::Tensor& tensor,
      Fn fn,
      int peer,
      OpType opType,
      const char* profilingTitle = nullptr);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> pointToPoint(
      at::Tensor& tensor,
      Fn fn,
      int peer,
      OpType opType,
      PreProcess pre,
      PostProcess post,
      const char* profilingTitle);

  c10::intrusive_ptr<Work> allreduce_impl(
      at::Tensor& tensor,
      const char* profilingTitle = "mccl:all_reduce",
      const AllreduceOptions& opts = AllreduceOptions());

  // Checks for MCCL errors on each of the communicators and returns an
  // appropriate exception_ptr (nullptr if no errors).
  static std::exception_ptr checkForMCCLErrorsInternal(
      std::shared_ptr<MCCLComm>& mcclComm);

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

  // Watchdog's inside loop.
  // Takes care of cleaning up completed work, and aborting upon failure or
  // timeout.
  void watchdogHandler();

  void runHookLoop();

  // Generates a prefix that is unique to this process group and rank, for
  // disambiguating logs
  std::string createLogPrefix() const;

  // Returns the unique prefix created in createLogPrefix
  const std::string& logPrefix() const;

  // Returns the global rank of the device. This function assumes that users
  // always create a default global process group(PG) which includes all
  // devices. It is called in the constructor of ProcessGroupMCCL, so it always
  // return the rank_ of the the very first PG created, aka, default global PG.
  const int& globalRank() const;

  // Returns the global ranks of a PG.
  const std::vector<uint64_t>& groupRanks() const;

  // Util function to assign timeout to each work.
  void assignTimeoutToWork(
      const c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>& work,
      const c10::intrusive_ptr<Options>& option);

  // Broadcast flight-recorder dump signal
  void broadcastDumpSignal();

  // A helper function to broadcast a signal (key) from a src rank to all other
  // ranks using the specified store.
  void broadcastSignal(
      c10::intrusive_ptr<Store>& store,
      const std::string& signal,
      int srcRank);

  // A helper function to get the src rank of a signal from the Store. This is
  // nonblocking function returning -1 if the signal is not available yet.
  int getSignalSrcRank(
      c10::intrusive_ptr<Store>& store,
      const std::string& signal);

  std::string dump_mccl_trace(
      bool includeCollectives,
      bool includeStackTraces,
      bool onlyActive);

 protected:
  // Function that runs as part of a separate thread aside from watchdog
  // thread because we need to check the heartbeat from watchdog thread
  // so that when we get stuck in some MCCL/MUSA calls,
  // we can dump the debugging information and abort the process.
  virtual void heartbeatMonitor();

  // Function that directly trigger std::abort so that the whole process
  // gets terminated.
  virtual void terminateProcess(const std::string& errMsg);

  // A helper function to wait for a future to complete or timeout.
  // Returns true if the future completes before timeout, false otherwise.
  bool waitForFutureOrTimeout(
      std::future<bool>& fut,
      const std::chrono::milliseconds& timeOutMilSec,
      const std::string& futDescription,
      ::c10d::C10dLoggingData& debugLog,
      bool throwException = false);

  std::string getMCCLWatchdogTimeoutErrorMsg(const std::string& extraMsg);

  std::string getMCCLWatchdogTimeoutExitMsg(const std::string& exitReason);

  void checkAndSetRemoteError();

  // A helper function to guess the device id of the current rank, based on
  // bounded device or used device. Do not use this function if you already know
  // the device id to operate on.
  c10::DeviceIndex guessDeviceId() const;

  static const int64_t kWatchdogThreadSleepMillis;
  static const int64_t kWorkCleanupThreadSleepMillis;
  static const int64_t kHeartBeatThreadSleepMillis;

  // The store is used to broadcast the MCCL unique ID of rank 0.
  c10::intrusive_ptr<Store> store_;

  bool storeError_{false};

  // Reference to the store without prefix so that keys are same across all
  // ProcessGroup MCCL instances and (key, value) pairs written to the store are
  // global.
  c10::intrusive_ptr<Store> globalStore_;

  // The lock which protects the write/read of
  // ephemeralTimeoutActive_/ephemeralTimeoutInflight_.
  // TODO(fduwjj): We need to have an audit on all mutexes we are adding here.
  // And consolidate them if possible.
  std::mutex mtxTimeoutExtension_;

  // The ephemeral timeout added on top of existing timeout for works issued
  // before first work finishes.
  std::chrono::milliseconds ephemeralTimeoutActive_ =
      std::chrono::milliseconds(0);

  // The ephemeral timeout addition which has been already applied to work.
  std::chrono::milliseconds ephemeralTimeoutInflight_ =
      std::chrono::milliseconds(0);

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
  std::unordered_map<std::string, std::shared_ptr<MCCLComm>> devMCCLCommMap_;

  // The MCCL communicators currently in process of being initialized.
  std::unordered_map<std::string, std::shared_ptr<MCCLComm>>
      inInitializationCommMap_;

  // Mutex to guard maps like devMCCLCommMap_ and mcclIdToCommMap_.
  std::mutex mutex_;

  // Heartbeat of watchdog thread.
  std::atomic_uint64_t heartbeat_{};

  // The time interval used for deciding whether there is no watchdog heartbeat.
  int heartbeatTimeoutInSec_{60 * 8}; // 8 min

  // timeout for the dump to finish.
  int waitTimeoutDumpInMilSec_;

  // promise to coordinate flight recorder dump.
  std::promise<void> promiseFlightRecorderDump_;

  // Interval of check coordinated signals in ProcessGroupMCCL from other ranks
  // e.g., trigger the dump of the debugging info for timeout when notified.
  int coordCheckIntervalMilSec_;

  // Size of ring buffer where we store MCCL Traces for debugging.
  int traceBufferSize_;

  // We gate the heartbeat monitor thread so that we can roll it out gradually.
  std::atomic<bool> monitorThreadEnabled_{};

  // We gate the cudaEventCache so that we can roll it out gradually.
  std::atomic<bool> musaEventCacheEnabled_{};

  // Monitor thread which checks the heartbeat of Watchdog thread.
  // If the monitor thread finds there is no heartbeat, it will dump debug info
  // and then kill the watchdog thread to avoid hang.
  std::thread mcclHeartbeatMonitorThread_;

  // Watchdog thread which looks for errors on the cached MCCL communicators.
  std::thread mcclCommWatchdogThread_;

  std::thread onCompletionHookThread_;

  // Whether or not we should terminate the watchdog and workCleanup threads.
  std::atomic<bool> terminateProcessGroup_;

  // Whether or not we should terminate the heartbeat monitoring threads.
  std::atomic<bool> terminateHeartbeatMonitorThread_;

  // Whether there are hooks pending to be fired
  std::atomic<bool> hasPendingHooks_{};

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

  // Mutex to Guard workMetaList_
  std::mutex workMetaListMutex_;

  // Mutex to Guard monitorWakeUpCV_
  std::mutex monitorMutex_;

  bool writeDebugInfo_ = false;

  // Condition Variable for timeout thread sleep
  std::condition_variable workMetaListCV_;

  // Condition Variable for monitor thread to wake up early
  std::condition_variable monitorWakeUpCV_;

  // Vector to Store WorkMCCL pointers
  std::list<ProcessGroupMCCL::WorkMCCL> workMetaList_;

  std::chrono::time_point<std::chrono::steady_clock> lastWorkListUpdateTime_;

  // Mutex to Guard workMetaList_
  std::mutex completedWorkListMutex_;

  // Condition Variable for watchdog thread sleep
  std::condition_variable completedWorkListCV_;

  std::list<ProcessGroupMCCL::WorkMCCL> completedWorkList_;

  // Add Work Pointer to workVector
  void workEnqueue(c10::intrusive_ptr<ProcessGroupMCCL::WorkMCCL>);

  // The MUSA streams used by MCCL kernels
  std::unordered_map<std::string, at::musa::MUSAStream> mcclStreams_;

  // The MUSA events used to sync MCCL streams
  std::unordered_map<std::string, at::musa::MUSAEvent> mcclEvents_;

  // Device Indexes used for all collectives in this group
  std::set<c10::DeviceIndex> usedDeviceIdxs_;

  // Flag to denote if a coalescing groupStart/groupEnd block is active
  int coalescing_state_ = 0;

  // Stores device indexes for all collectives run inside a coalescing block
  at::Device coalescedDevice_ = at::Device(at::DeviceType::PrivateUse1);

  // Stores communicators for all collectives run inside a coalescing block
  std::shared_ptr<MCCLComm> coalescedComm_ = nullptr;

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

  // Whether or not to hook the cache allocator to register all allocated
  // tensors
  bool useTensorRegisterAllocatorHook_ = false;

  // Whether or not the workCleanupThread is used to perform async error
  // handling.
  ErrorHandlingMode asyncErrorHandling_ = NoHandling;

  ErrorType error_ = ErrorType::SUCCESS;

  std::mutex errorMutex_;

  // Whether or not to enable timeout root cause analysis.
  bool desyncDebug_;
  DesyncDebugger desyncDebugger_;

  // Whether or not to dump debug info on exception including both watchdog
  // timeout and mccl errors.
  bool dumpOnTimeoutOrEx_;

  // Whether or not to propagate detected errors to all ranks in the same PG
  // through TCPStore.
  bool propagatePgError_;

  // Whether or not to sleep after an exception is thrown in the watchdog.
  bool sleepAfterException_{};

  // Whether or not to enable nan check for input tensors to collectives.
  bool enableNanCheck_;

  // Whether or not to print C++ stack traces to logs on unclean shutdown.
  bool logCppStackOnUncleanShutdown_;

  // Whether or not to create start MUSAEvent and enable timing for start
  // and end events. Note that enableTiming_ is always true if desyncDebug_
  // is set to true.
  std::atomic<bool> enableTiming_{};

  // Flag to enable the print of hash value of input/output of collectives for
  // verification.
  std::atomic<bool> enableCollecticeHashDebug_{};

  // Whether or not TORCH_MCCL_AVOID_RECORD_STREAMS was set
  bool avoidRecordStreams_ = false;

  // Whether the MCCL watchdog should rethrow MUSA errors.
  bool rethrowMUSAErrors_ = false;

  // The number of active mcclGroupStart() calls. This counter will be increased
  // by 1 when mcclGroupStart() is called and decreased by 1 when mcclGroupEnd()
  // is called.
  static thread_local uint64_t mcclActiveGroupCounter_;

  // Counting for the sequential number of MCCL collective call.
  // (specifically, how many actual kernels we launched, which differs from
  // op_id_ when coalescing is enabled)
  uint64_t seqCollective_{0};

  // Counting for the sequential number of MCCL P2P calls.
  uint64_t seqP2P_{0};

  // Incrementing counter for logical operations (collective or p2p) issued on
  // the ProcessGroup
  uint64_t op_id_{0};

  std::exception_ptr watchDogException_ = nullptr;

  // The number of ProcessGroupMCCL created on the current rank.
  size_t local_id_{0};

  std::string logPrefix_;

  // a placeholder for intraNodeComm
  int intraNodeComm_{-1};

  // Number of devices on this node.
  int localDeviceCount_{0};

  // the flag of heartbeat monitor for mccl status
  bool mcclHeartBeatMonitor_{false};

  std::shared_ptr<ProcessGroupStatus> pgStatus_ =
      std::make_shared<ProcessGroupStatus>();

  // Internal cached value: use MCCL non-blocking API mode or not.
  // Use `useNonblocking()` method instead of accessing this variable directly.
  std::optional<bool> useNonblocking_{std::nullopt};

  // Communication-optimized memory pool associated with this PG
  std::unique_ptr<c10::musa::MemPool> memPool_ = nullptr;

}; // class ProcessGroupMCCL

// Similar to get_cpp_trace_dumper, this stores a function defined in
// torch-python layer that lets us check whether the GIL can be acquired,
// helpful for instrumenting in cases where a hang was observed.
typedef bool (*gil_checker_t)();

TORCH_API gil_checker_t& get_gil_checker();
} // namespace c10d

void registerProcessGroupMCCL(PyObject* mod);
//#endif // USE_C10D_MCCL

#endif // TORCH_MUSA_CSRC_DISTRIBUTED_PROCESSGROUPMCCL_H_
