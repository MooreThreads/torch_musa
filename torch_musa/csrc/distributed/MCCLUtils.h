#ifndef TORCH_MUSA_CSRC_DISTRIBUTED_MCCLUTILS_H_
#define TORCH_MUSA_CSRC_DISTRIBUTED_MCCLUTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <mutex>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <mccl.h>
#include <torch/csrc/distributed/c10d/TraceUtils.h>

#include "torch_musa/csrc/core/MUSAEvent.h"

// Macro to throw on a non-successful MCCL return value.
#define C10D_MCCL_CHECK(cmd, failureReason)                                   \
  do {                                                                        \
    mcclResult_t result = cmd;                                                \
    if (result != mcclSuccess) {                                              \
      std::string err = "MCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + mcclGetErrorWithVersion(result) + \
          "\n" + getMcclErrorDetailStr(result, failureReason);                \
      TORCH_CHECK(false, err);                                                \
    }                                                                         \
  } while (0)

// Macro to print and abort on a non-successful MCCL return value.
#define C10D_MCCL_ASSERT(cmd)                            \
  do {                                                   \
    mcclResult_t result = cmd;                           \
    if (result != mcclSuccess) {                         \
      std::string err = mcclGetErrorWithVersion(result); \
      fprintf(                                           \
          stderr,                                        \
          "[Error] MCCL error in: %s:%d, %s\n",          \
          __FILE__,                                      \
          __LINE__,                                      \
          err.c_str());                                  \
      abort();                                           \
    }                                                    \
  } while (0)

#define DEFINE_CONSTANT(name, value) \
  static c10::IValue name = value;   \
  static std::string name##_str = value;
// Update whenever changing contents or formatting of the dump
// (minor when adding fields, major when changing existing fields)
// Also update both JSON and Pickle dumps to make use of the newly defined
// field(s).
DEFINE_CONSTANT(version_val, "2.4")
DEFINE_CONSTANT(entries_key, "entries");
DEFINE_CONSTANT(mccl_comm_key, "mccl_comm_state");
DEFINE_CONSTANT(version_key, "version");
DEFINE_CONSTANT(pg_config_key, "pg_config");
DEFINE_CONSTANT(pg_status_key, "pg_status");
DEFINE_CONSTANT(record_id_key, "record_id");
DEFINE_CONSTANT(pg_id_key, "pg_id");
DEFINE_CONSTANT(pg_name_key, "process_group");
DEFINE_CONSTANT(collective_seq_id_key, "collective_seq_id");
DEFINE_CONSTANT(p2p_seq_id_key, "p2p_seq_id");
DEFINE_CONSTANT(is_p2p_key, "is_p2p");
DEFINE_CONSTANT(op_id_key, "op_id");
DEFINE_CONSTANT(profiling_name_key, "profiling_name");
DEFINE_CONSTANT(input_sizes_key, "input_sizes");
DEFINE_CONSTANT(input_dtypes_key, "input_dtypes");
DEFINE_CONSTANT(output_sizes_key, "output_sizes");
DEFINE_CONSTANT(output_dtypes_key, "output_dtypes");
DEFINE_CONSTANT(time_created_key, "time_created_ns");
DEFINE_CONSTANT(duration_key, "duration_ms");
DEFINE_CONSTANT(timeout_key, "timeout_ms");
DEFINE_CONSTANT(frames_key, "frames");
DEFINE_CONSTANT(state_key, "state");
DEFINE_CONSTANT(line_key, "line");
DEFINE_CONSTANT(name_key, "name");
DEFINE_CONSTANT(filename_key, "filename");
DEFINE_CONSTANT(retired_key, "retired");
DEFINE_CONSTANT(time_discovered_started_key, "time_discovered_started_ns");
DEFINE_CONSTANT(time_discovered_completed_key, "time_discovered_completed_ns");
DEFINE_CONSTANT(completed_state, "completed");
DEFINE_CONSTANT(scheduled_state, "scheduled");
DEFINE_CONSTANT(started_state, "started");
#undef DEFINE_CONSTANT

namespace c10d {

std::string getMcclVersion();
std::string mcclGetErrorWithVersion(mcclResult_t error);

std::string getMcclErrorDetailStr(
    mcclResult_t error,
    c10::optional<std::string> processGroupFailureReason = c10::nullopt);

// RAII wrapper for MCCL communicator
class MCCLComm {
  using MutexType = std::recursive_mutex;
  using LockType = std::unique_lock<MutexType>;

 public:
  explicit MCCLComm(mcclComm_t mcclComm);

  MCCLComm() = default;

  ~MCCLComm() noexcept;

  static std::shared_ptr<MCCLComm> create(
      int numRanks,
      int rank,
      mcclUniqueId commId,
      at::DeviceIndex deviceIndex);

  std::unordered_map<std::string, std::string> mcclCommDump();

  mcclUniqueId getMcclId();

  // Must not be copyable
  MCCLComm(const MCCLComm&) = delete;
  MCCLComm& operator=(const MCCLComm&) = delete;

  // Do not support move assignment as there is no valid use case
  MCCLComm& operator=(MCCLComm&& other) = delete;

  // Move constructable
  // NOLINTNEXTLINE(*-noexcept-move-*)
  MCCLComm(MCCLComm&& other);

  mcclComm_t getMcclComm();

  void waitReady(bool longInterval);

  c10::optional<std::string> getMcclCommFailureReason() const;

  void abort(std::optional<std::string> commFailureReason = std::nullopt);

  void finalize();

  // Destroy a communicator. This is a blocking function.
  void destroy();

  bool isInitialized() const;

  bool isAborted() const;

  uint64_t getCommSplitCounter() const;

  mcclResult_t checkForMcclError();

  mcclResult_t registerSegment(
      void* ptr,
      size_t size,
      bool errorOnRereg = true);

  mcclResult_t deregisterSegment(void* ptr);

  std::string repr() const;

  friend class ProcessGroupMCCL;

 protected:
  // Unique mccl_id for this communicator.
  mcclUniqueId mcclId_{};
  bool aborted_{false};
  uint64_t mcclCommSplitCounter_{0};
  mcclResult_t mcclAsyncErr_{mcclSuccess};
  mutable MutexType mutex_;
  // Rank that this communicator corresponds to.
  int rank_;
  // Optional reason for communicator failure, provided by ProcessGroupMCCL for
  // better error messaging.
  std::optional<std::string> commFailureReason_{};
  bool initialized_{false};
  // Whether this communicator is using nonblocking mode. Recorded during comm
  // creation or split. For safety, we give a default value of true (more
  // protection).
  bool nonBlocking_{true};
  // Device index for which the MCCL comm is created
  at::DeviceIndex deviceIndex_{-1};

 private:
  mcclComm_t mcclComm_{nullptr};
};

class DebugInfoWriter {
 public:
  virtual ~DebugInfoWriter() = default;
  virtual void write(const std::string& trace);
  static DebugInfoWriter& getWriter(int rank);
  static void registerWriter(std::unique_ptr<DebugInfoWriter> writer);
  virtual std::string getWriterTarget() {
    return filename_;
  }

 protected:
  DebugInfoWriter(std::string namePrefix, int rank) {
    filename_ = c10::str(namePrefix, rank);
  }
  std::string filename_;

 private:
  static std::unique_ptr<DebugInfoWriter> writer_;
  static std::atomic<bool> hasWriterRegistered_;
};

/* Helper used by work::getDuration() and mccl flight recorder */
float getDurationFromEvent(
    at::musa::MUSAEvent& mcclStartEvent,
    at::musa::MUSAEvent& mcclEndEvent);

struct FlightRecorder {
  static FlightRecorder* get() {
    // intentionally leak on exit
    // because this will hold python state that may get destructed
    static FlightRecorder* instance = new FlightRecorder();
    return instance;
  }
  FlightRecorder() {
    max_entries_ = getCvarInt({"TORCH_MCCL_TRACE_BUFFER_SIZE"}, 0);
    capture_cpp_stack_ = getCvarBool({"TORCH_MCCL_TRACE_CPP_STACK"}, false);
    enabled_ = max_entries_ > 0;
  }
  using Event = at::musa::MUSAEvent;
  struct Entry {
    size_t id_; // incremented id in the trace buffer
                // used to figure out where in the circular entries
                // buffer this entry will be located to
                // update state information
    size_t pg_id_;
    std::tuple<std::string, std::string> pg_name_; // <group_name, group_desc>

    // collective_seq_id and p2p_seq_id refer to actual kernel launches (e.g. 1
    // per coalesced group).
    // collective_seq_id only increments for true collective operations (over
    // all ranks in the group). p2p_seq_id only increments over non-collective
    // operations in the group. op_id refers to logical operations (e.g. one per
    // op inside coalesced group)
    size_t collective_seq_id_;
    size_t p2p_seq_id_;
    size_t op_id_;
    std::string profiling_name_;

    std::shared_ptr<torch::CapturedTraceback> traceback_;
    // we borrow pointers to start_ and end_ so we can query the state
    // on reporting. However, once the event is completed, the call
    // to `complete` will clear these.
    Event *start_, *end_;

    // timestamp when the entry was created, likely close to the time the work
    // was 'enqueued'- not necessarily started
    c10::time_t time_created_;

    // configured timeout for this entry
    c10::time_t timeout_ms_;

    // Is this a P2P event?
    bool isP2P_;

    std::optional<float> duration_;

    // timestamp when our CPU threads discovered that the kernel started.
    // will always be _after_ it actually started, and can be very late
    // if the watchdog thread got stuck on MUSA APIs.
    std::optional<c10::time_t> time_discovered_started_;

    // timestamp when our CPU threads discovered that the kernel completed.
    // will always be _after_ it actually complated, and can be the same time
    // as the discovery of the start if the watchdog thread is stuck on MUSA
    // APIs
    std::optional<c10::time_t> time_discovered_completed_;

    // size information for input/output tensors
    c10::SmallVector<int64_t, 4> input_dims_;
    std::vector<c10::ScalarType> input_dtypes_;
    c10::SmallVector<int64_t, 4> output_dims_;
    std::vector<c10::ScalarType> output_dtypes_;
    c10::SmallVector<int64_t, 8> sizes_; // flattened from inputs, outputs
    bool retired_ = false; // is this work entry no longer in the workMetaList_?
                           // a retired but not completed event has timed out

    // Returns the traceback of current entry, in string form.
    std::string getTraceback();
  };

  bool enabled_ = false;
  bool capture_cpp_stack_ = false;
  std::mutex mutex_;
  std::vector<Entry> entries_;
  size_t max_entries_ = 0;
  size_t next_ = 0;
  size_t id_ = 0;
  std::map<size_t, std::shared_ptr<ProcessGroupStatus>> all_pg_status_ = {};
  std::map<std::tuple<std::string, std::string>, std::vector<uint64_t>>
      pg_name_to_ranks_ = {};

  std::optional<size_t> record(
      size_t pg_id,
      const std::tuple<std::string, std::string>& pg_name,
      size_t collective_seq_id,
      size_t p2p_seq_id,
      size_t op_id,
      std::string profiling_name,
      const std::vector<at::Tensor>& inputs,
      const std::vector<at::Tensor>& outputs,
      Event* start,
      Event* end,
      std::chrono::milliseconds timeout_ms,
      std::shared_ptr<ProcessGroupStatus> pg_status,
      bool isP2P);

  void record_pg_ranks(
      const std::tuple<std::string, std::string>& pg_name,
      std::vector<uint64_t> ranks);

  void update_state(Entry& r);

  std::vector<Entry> dump_entries();

  // Returns the entry with the given id, if it exists. Otherwise, returns
  // std::nullopt.
  std::optional<Entry> getEntry(std::optional<size_t> id);

  /*
  Mark an Event as completed and free its events.
  This is called by the watchdog thread, and is asynchronous from the
  perspective of the main thread.
  compute_duration defaults to true since retire_id is only called in the
  watchdog thread, which is currently a place we call musa APIs which may hang,
  but care should be taken to avoid computing duration in any function that must
  never hang. (timing must also be enabled for compute_duration - see
  TORCH_MCCL_ENABLE_TIMING).
  */
  void retire_id(std::optional<size_t> id, bool compute_duration = true);

  const c10::List<c10::IValue> getCollectiveTrace(
      bool includeStacktraces,
      bool onlyActive);

  // dump pg_entries
  const c10::Dict<c10::IValue, c10::IValue> getPgConfig();

  const std::map<std::string, std::map<std::string, std::string>>
  getPgConfigJson();

  // dump pg_status
  const c10::Dict<c10::IValue, c10::IValue> getPgStatus();

  const std::map<std::string, std::map<std::string, std::string>>
  getPgStatusJson();

  // std::string dump_json(
  //     const std::optional<std::unordered_map<
  //         std::string,
  //         std::unordered_map<std::string, std::string>>>& mcclDumpMap,
  //     bool includeCollectives,
  //     bool onlyActive);

  // dump all collectives + mcclDumpMap
  std::string dump(
      const std::optional<std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::string>>>& mcclDumpMap,
      bool includeCollectives,
      bool includeStackTraces,
      bool onlyActive);
};

// Helper that automatically cleans up premul sums.
struct mcclRedOpRAII {
  mcclRedOpRAII() = default;
  mcclRedOpRAII(mcclRedOp_t op) : op_(op) {}
  mcclRedOpRAII(mcclRedOp_t op, mcclComm_t comm)
      : op_(op), comm_(comm), premul_sum_(true) {}
  mcclRedOpRAII(const mcclRedOpRAII&) = delete;
  mcclRedOpRAII& operator=(const mcclRedOpRAII&) = delete;
  mcclRedOpRAII(mcclRedOpRAII&& tmp) : mcclRedOpRAII() {
    std::swap(tmp.op_, this->op_);
    std::swap(tmp.comm_, this->comm_);
    std::swap(tmp.premul_sum_, this->premul_sum_);
  }
  ~mcclRedOpRAII() {
    if (premul_sum_) {
      mcclRedOpDestroy(op_, comm_);
    }
  }
  operator mcclRedOp_t() const {
    return op_;
  }
  mcclRedOp_t op_;
  mcclComm_t comm_;
  bool premul_sum_ = false;
};

struct MCCLTraceBuffer {
  static MCCLTraceBuffer* get() {
    // intentionally leak on exit
    // because this will hold python state that may get destructed
    static MCCLTraceBuffer* instance = new MCCLTraceBuffer();
    return instance;
  }
  MCCLTraceBuffer() {
    max_entries_ = getCvarInt({"TORCH_MCCL_TRACE_BUFFER_SIZE"}, 0);
    capture_cpp_stack_ = getCvarBool({"TORCH_MCCL_TRACE_CPP_STACK"}, false);
    enabled_ = max_entries_ > 0;
  }
  using Event = at::musa::MUSAEvent;
  struct Entry {
    size_t id_; // incremented id in the trace buffer
                // used to figure out where in the circular entries
                // buffer this entry will be located to
                // update state information
    size_t pg_id_;
    std::tuple<std::string, std::string> pg_name_; // <group_name, group_desc>

    // collective_seq_id and p2p_seq_id refer to actual kernel launches (e.g. 1
    // per coalesced group).
    // collective_seq_id only increments for true collective operations (over
    // all ranks in the group). p2p_seq_id only increments over non-collective
    // operations in the group. op_id refers to logical operations (e.g. one per
    // op inside coalesced group)
    size_t collective_seq_id_;
    size_t p2p_seq_id_;
    size_t op_id_;
    std::string profiling_name_;

    std::shared_ptr<torch::CapturedTraceback> traceback_;

    // we borrow pointers to start_ and end_ so we can query the state
    // on reporting. However, once the event is completed, the call
    // to `complete` will clear these.
    Event *start_, *end_;

    // timestamp when the entry was created, likely close to the time the work
    // was 'enqueued'- not necessarily started
    c10::time_t time_created_;

    // configured timeout for this entry
    c10::time_t timeout_ms_;

    // Is this a P2P event?
    bool isP2P_;

    std::optional<float> duration_;

    // timestamp when our CPU threads discovered that the kernel started.
    // will always be _after_ it actually started, and can be very late
    // if the watchdog thread got stuck on MUSA APIs.
    std::optional<c10::time_t> time_discovered_started_;

    // timestamp when our CPU threads discovered that the kernel completed.
    // will always be _after_ it actually complated, and can be the same time
    // as the discovery of the start if the watchdog thread is stuck on MUSA
    // APIs
    std::optional<c10::time_t> time_discovered_completed_;

    // size information for input/output tensors
    c10::SmallVector<int, 4> input_dims_;
    std::vector<c10::ScalarType> input_dtypes_;
    c10::SmallVector<int, 4> output_dims_;
    std::vector<c10::ScalarType> output_dtypes_;
    c10::SmallVector<int64_t, 8> sizes_; // flattened from inputs, outputs
    bool retired_ = false; // is this work entry no longer in the workMetaList_?
                           // a retired but not completed event has timed out
  };

  bool enabled_ = false;
  bool capture_cpp_stack_ = false;
  std::mutex mutex_;
  std::vector<Entry> entries_;
  size_t max_entries_ = 0;
  size_t next_ = 0;
  size_t id_ = 0;
  std::map<size_t, std::shared_ptr<ProcessGroupStatus>> all_pg_status_ = {};
  std::map<std::tuple<std::string, std::string>, std::vector<uint64_t>>
      pg_name_to_ranks_ = {};

  void update_state(Entry& r);

  std::vector<Entry> dump_entries();

  const c10::List<c10::IValue> getCollectiveTrace(
      bool includeStacktraces,
      bool onlyActive);

  // dump pg_entries
  const c10::Dict<c10::IValue, c10::IValue> getPgConfig();

  // dump pg_status
  const c10::Dict<c10::IValue, c10::IValue> getPgStatus();

  // dump all collectives + mcclDumpMap
  std::string dump(
      const std::optional<std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::string>>>& mcclDumpMap,
      bool includeCollectives,
      bool includeStackTraces,
      bool onlyActive);

  // TODO(jihong.zhong): finish below member functions before 2025.6.30

  // std::optional<size_t> record(
  //     size_t pg_id,
  //     const std::tuple<std::string, std::string>& pg_name,
  //     size_t collective_seq_id,
  //     size_t p2p_seq_id,
  //     size_t op_id,
  //     std::string profiling_name,
  //     const std::vector<at::Tensor>& inputs,
  //     const std::vector<at::Tensor>& outputs,
  //     Event* start,
  //     Event* end,
  //     std::chrono::milliseconds timeout_ms,
  //     std::shared_ptr<ProcessGroupStatus> pg_status,
  //     bool isP2P);

  // void record_pg_ranks(
  //     const std::tuple<std::string, std::string>& pg_name,
  //     std::vector<uint64_t> ranks);

  // /*
  // Mark an Event as completed and free its events.
  // This is called by the watchdog thread, and is asynchronous from the
  // perspective of the main thread.
  // compute_duration defaults to true since retire_id is only called in the
  // watchdog thread, which is currently a place we call musa APIs which may
  // hang, but care should be taken to avoid computing duration in any function
  // that must never hang. (timing must also be enabled for compute_duration -
  // see TORCH_MCCL_ENABLE_TIMING).
  // */
  // void retire_id(std::optional<size_t> id, bool compute_duration = true);

  // const std::map<std::string, std::map<std::string, std::string>>
  // getPgConfigJson();

  // const std::map<std::string, std::map<std::string, std::string>>
  // getPgStatusJson();

  // std::string dump_json(
  //     const std::optional<std::unordered_map<
  //         std::string,
  //         std::unordered_map<std::string, std::string>>>& mcclDumpMap,
  //     bool includeCollectives,
  //     bool onlyActive);
};

} // namespace c10d

#endif // TORCH_MUSA_CSRC_DISTRIBUTED_MCCLUTILS_H_