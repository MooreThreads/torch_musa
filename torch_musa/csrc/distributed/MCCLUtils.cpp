#include "torch_musa/csrc/distributed/MCCLUtils.h"
#include <c10/util/CallOnce.h>

#include <fstream>
#include <mutex>
#include <vector>

namespace c10d {

MCCLComm::MCCLComm(mcclComm_t mcclComm) : mcclComm_(mcclComm) {}

MCCLComm::~MCCLComm() noexcept {
  LockType lock(mutex_);
  if (mcclComm_ && initialized_ && !aborted_) {
    // TORCH_WARN_ONCE(
    //     "WARNING: MCCL communicator hasn't been destroyed. This may cause "
    //     "memory leaks. To avoid the risk, you can call
    //     `destroy_process_group` " "during normal exit or
    //     `_abort_process_group` when handling failures.")
    C10D_MCCL_ASSERT(::mcclCommAbort(mcclComm_));
  }
}

// NOLINTNEXTLINE(*-noexcept-move-*)
MCCLComm::MCCLComm(MCCLComm&& other) {
  // Using other's lock, as it reads other's states
  // Can not use this.mutex_, as this object is being constructed.
  LockType lock(other.mutex_);
  std::swap(mcclComm_, other.mcclComm_);
  std::swap(aborted_, other.aborted_);
  std::swap(mcclAsyncErr_, other.mcclAsyncErr_);
  std::swap(initialized_, other.initialized_);
  std::swap(nonBlocking_, other.nonBlocking_);
  std::swap(deviceIndex_, other.deviceIndex_);
}

mcclUniqueId MCCLComm::getMcclId() {
  return mcclId_;
}

std::shared_ptr<MCCLComm> MCCLComm::create(
    int numRanks,
    int rank,
    mcclUniqueId commId,
    at::DeviceIndex deviceIndex) {
  at::musa::OptionalMUSAGuard gpuGuard(deviceIndex);
  auto comm = std::make_shared<MCCLComm>();
  C10D_MCCL_CHECK(
      mcclCommInitRank(&(comm->mcclComm_), numRanks, commId, rank),
      std::nullopt);
  comm->mcclId_ = commId;
  comm->rank_ = rank;
  comm->deviceIndex_ = deviceIndex;
  comm->initialized_ = true;
  // Old style comm is always blocking.
  comm->nonBlocking_ = false;
  return comm;
}

mcclComm_t MCCLComm::getMcclComm() {
  LockType lock(mutex_);
  if (aborted_) {
    auto commFailureMsg = commFailureReason_ != c10::nullopt
        ? c10::str(" Original reason for failure was: ", *commFailureReason_)
        : "";
    TORCH_CHECK(
        false,
        c10::str(
            "MCCL communicator was aborted on rank ",
            rank_,
            ". ",
            commFailureMsg));
  }
  if (!initialized_) {
    // TODO: see if we can consolidate other `initialized_` flipping here.
    // Maintaining it elsewhere is some work.
    initialized_ = true;
    LOG(INFO) << "Rank " << rank_ << ": MCCL communicator " << repr()
              << " is initialized.";
  }
  return mcclComm_;
}

// Wait for the communicator to be ready. This is a blocking function.
// Arguments:
//   longInterval: if true, wait with sleep of an interval; otherwise, wait
//   with `sched_yield` which is faster (but acquires CPU more frequently).
void MCCLComm::waitReady(bool longInterval) {
  LockType lock(mutex_);
  if (aborted_)
    return;
  // // If timeout is reached, throw an exception.
  // if (longInterval) {
  //   C10D_MCCL_CHECK_TIMEOUT_SLEEP(mcclInProgress, mcclComm_, std::nullopt);
  // } else {
  //   C10D_MCCL_CHECK_TIMEOUT(mcclInProgress, mcclComm_, std::nullopt);
  // }
}

std::optional<std::string> MCCLComm::getMcclCommFailureReason() const {
  LockType lock(mutex_);
  return commFailureReason_;
}

void MCCLComm::finalize() {
  LockType lock(mutex_);
  if (aborted_) {
    LOG(INFO) << "Rank " << rank_
              << ": MCCL communicator already Invalidated. Skip finalize.";
    return;
  }
  // at::musa::OptionalMUSAGuard gpuGuard(deviceIndex_);
  // auto comm = getMcclComm();
  // C10D_MCCL_CHECK_NONBLOCKING(mcclCommFinalize(comm), std::nullopt);
}

void MCCLComm::destroy() {
  LockType lock(mutex_);
  if (aborted_) {
    LOG(INFO) << "Rank " << rank_
              << ": MCCL communicator already Invalidated. Skip destroy.";
    return;
  }
  at::musa::OptionalMUSAGuard gpuGuard(deviceIndex_);
  auto comm = getMcclComm();
  C10D_MCCL_CHECK(mcclCommDestroy(comm), std::nullopt);
  // Poison future getMcclComm
  aborted_ = true;
}

void MCCLComm::abort(std::optional<std::string> commFailureReason) {
  LockType lock(mutex_);
  at::musa::OptionalMUSAGuard gpuGuard(deviceIndex_);
  if (aborted_ && !initialized_) {
    // Should not abort twice.
    return;
  }
  // Set true failure reason if provided by ProcessGroupMCCL (e.g. work
  // timeout)
  commFailureReason_ = commFailureReason;
  LOG(INFO) << "Aborting mcclComm_ " << mcclComm_ << " with reason: "
            << (commFailureReason ? *commFailureReason
                                  : "No abort reason provided.");

  C10D_MCCL_CHECK(::mcclCommAbort(mcclComm_), commFailureReason_);

  aborted_ = true;
  mcclComm_ = nullptr;

  // Set an appropriate error so that we avoid using the communicator.
  if (mcclAsyncErr_ == mcclSuccess) {
    mcclAsyncErr_ = mcclSystemError;
  }
}

bool MCCLComm::isInitialized() const {
  LockType lock(mutex_);
  return initialized_;
}

bool MCCLComm::isAborted() const {
  LockType lock(mutex_);
  return aborted_;
}

uint64_t MCCLComm::getCommSplitCounter() const {
  return mcclCommSplitCounter_;
}

mcclResult_t MCCLComm::checkForMcclError() {
  LockType lock(mutex_);
  if (mcclAsyncErr_ != mcclSuccess) {
    return mcclAsyncErr_;
  }
  C10D_MCCL_CHECK(
      mcclCommGetAsyncError(mcclComm_, &mcclAsyncErr_), commFailureReason_);
  return mcclAsyncErr_;
}

mcclResult_t MCCLComm::registerSegment(
    void* ptr,
    size_t size,
    bool errorOnRereg /*=true*/) {
  LockType lock(mutex_);
  return mcclInvalidUsage;
}

mcclResult_t MCCLComm::deregisterSegment(void* ptr) {
  LockType lock(mutex_);
  return mcclInvalidUsage;
}

std::string MCCLComm::repr() const {
  return c10::str((void*)mcclComm_);
}

std::unordered_map<std::string, std::string> MCCLComm::mcclCommDump() {
  std::unordered_map<std::string, std::string> dump;
  if (isAborted()) {
    LOG(INFO) << "Communicator was aborted before trying to dump its state.";
    return dump;
  }
  // TODO(jihong.zhong): finish below api in mccl
  // C10D_MCCL_CHECK(::mcclCommDump(mcclComm_, dump), std::nullopt);
  return dump;
}

std::string getMcclVersion() {
  static c10::once_flag mcclGetVersionFlag;
  static std::string versionString;
  c10::call_once(mcclGetVersionFlag, []() {
    int version = 0;
    mcclResult_t status = mcclGetVersion(&version);
    if (status != mcclSuccess || version < 100) {
      versionString = "Unknown MCCL version";
    } else {
      const int majorBase = version < 2900 ? 1000 : 10000;
      const int minorBase = 100;
      int mcclMajor = version / majorBase;
      int mcclMinor = (version % majorBase) / minorBase;
      int mcclPatch = version % minorBase;
      versionString = std::to_string(mcclMajor) + "." +
          std::to_string(mcclMinor) + "." + std::to_string(mcclPatch);
    }
  });
  return versionString;
}

std::string mcclGetErrorWithVersion(mcclResult_t error) {
  return std::string(mcclGetErrorString(error)) + ", MCCL version " +
      std::string(getMcclVersion());
}

std::string getMcclErrorDetailStr(
    mcclResult_t error,
    c10::optional<std::string> processGroupFailureReason) {
  if (processGroupFailureReason != c10::nullopt) {
    return *processGroupFailureReason;
  }
  // TODO(yueran-tang): Complete Error info str in the future.
  return mcclGetErrorWithVersion(error);
}

const c10::List<c10::IValue> MCCLTraceBuffer::getCollectiveTrace(
    bool includeStacktraces,
    bool onlyActive) {
  auto entries = new_list();
  // Entries are returned in the order they were recorded
  auto result = dump_entries();
  std::vector<torch::CapturedTraceback*> tracebacks;
  torch::SymbolizedTracebacks stracebacks;
  std::vector<c10::IValue> all_frames;
  if (includeStacktraces) {
    for (auto& e : result) {
      tracebacks.push_back(e.traceback_.get());
    }
    stracebacks = torch::symbolize(tracebacks);
    for (const auto& f : stracebacks.all_frames) {
      auto d = new_dict();
      d.insert(name_key, f.funcname);
      d.insert(filename_key, f.filename);
      d.insert(line_key, int64_t(f.lineno));
      all_frames.emplace_back(std::move(d));
    }
  }
  for (auto i : c10::irange(result.size())) {
    auto dict = new_dict();
    auto& e = result.at(i);
    // Skip completed events
    if (onlyActive && e.time_discovered_completed_.has_value()) {
      continue;
    }
    if (includeStacktraces) {
      auto& tb = stracebacks.tracebacks.at(i);
      auto frames = new_list();
      for (int64_t frame : tb) {
        frames.push_back(all_frames.at(frame));
      }
      dict.insert(frames_key, frames);
    }

    dict.insert(record_id_key, int64_t(e.id_));
    dict.insert(pg_id_key, int64_t(e.pg_id_));
    dict.insert(pg_name_key, e.pg_name_);
    dict.insert(collective_seq_id_key, int64_t(e.collective_seq_id_));
    dict.insert(p2p_seq_id_key, int64_t(e.p2p_seq_id_));
    dict.insert(op_id_key, int64_t(e.op_id_));
    dict.insert(profiling_name_key, e.profiling_name_);
    dict.insert(time_created_key, int64_t(e.time_created_));
    if (e.duration_) {
      dict.insert(duration_key, *e.duration_);
    }

    auto it = e.sizes_.begin();
    auto read_sizes = [&](const c10::SmallVector<int, 4>& dims) {
      auto sizes = new_list();
      for (auto dim : dims) {
        auto arg_sizes = new_list();
        for (C10_UNUSED auto i : c10::irange(dim)) {
          arg_sizes.push_back(*it++);
        }
        sizes.push_back(arg_sizes);
      }
      return sizes;
    };

    dict.insert(input_sizes_key, read_sizes(e.input_dims_));
    std::vector<std::string> input_dtypes_strs;
    input_dtypes_strs.reserve(e.input_dtypes_.size());
    for (const auto& input_dtype : e.input_dtypes_) {
      input_dtypes_strs.push_back(c10::toString(input_dtype));
    }
    dict.insert(input_dtypes_key, input_dtypes_strs);
    dict.insert(output_sizes_key, read_sizes(e.output_dims_));
    std::vector<std::string> output_dtypes_strs;
    output_dtypes_strs.reserve(e.output_dtypes_.size());
    for (const auto& output_dtype : e.output_dtypes_) {
      output_dtypes_strs.push_back(c10::toString(output_dtype));
    }
    dict.insert(output_dtypes_key, output_dtypes_strs);
    if (e.time_discovered_completed_.has_value()) {
      dict.insert(state_key, completed_state);
    } else if (e.time_discovered_started_.has_value()) {
      dict.insert(state_key, started_state);
    } else {
      dict.insert(state_key, scheduled_state);
    }

    dict.insert(
        time_discovered_started_key,
        e.time_discovered_started_.has_value()
            ? int64_t(*e.time_discovered_started_)
            : c10::IValue());
    dict.insert(
        time_discovered_completed_key,
        e.time_discovered_completed_.has_value()
            ? int64_t(*e.time_discovered_completed_)
            : c10::IValue());
    dict.insert(retired_key, e.retired_);
    dict.insert(timeout_key, e.timeout_ms_);
    dict.insert(is_p2p_key, e.isP2P_);

    entries.push_back(dict);
  }
  return entries;
}

const c10::Dict<c10::IValue, c10::IValue> MCCLTraceBuffer::getPgConfig() {
  auto pg_config = new_dict();
  for (const auto& [pg_name, ranks] : pg_name_to_ranks_) {
    auto pg_info = new_dict();
    pg_info.insert("name", std::get<0>(pg_name));
    pg_info.insert("desc", std::get<1>(pg_name));
    pg_info.insert("ranks", ranks_str(ranks));
    pg_config.insert(std::get<0>(pg_name), pg_info);
  }
  return pg_config;
}

const c10::Dict<c10::IValue, c10::IValue> MCCLTraceBuffer::getPgStatus() {
  auto all_pg_status = new_dict();
  for (const auto& [pg_id, status] : all_pg_status_) {
    auto pg_status = new_dict();
    pg_status.insert("last_enqueued_collective", status->lastEnqueuedSeq);
    pg_status.insert("last_started_collective", status->lastStartedSeq);
    pg_status.insert("last_completed_collective", status->lastCompletedSeq);
    all_pg_status.insert(std::to_string(pg_id), pg_status);
  }
  return all_pg_status;
}

std::string MCCLTraceBuffer::dump(
    const std::optional<std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::string>>>& mcclDumpMap,
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  auto result = new_dict();
  // common values
  result.insert(version_key, getMcclVersion().c_str());
  result.insert(pg_config_key, getPgConfig());
  result.insert(pg_status_key, getPgStatus());

  // collective trace
  if (includeCollectives) {
    result.insert(
        entries_key, getCollectiveTrace(includeStackTraces, onlyActive));
  }
  // convert mcclDumpMap into a dictionary
  auto per_comm_dict = new_dict();
  if (mcclDumpMap.has_value()) {
    for (const auto& [mcclId, mcclDump] : mcclDumpMap.value()) {
      auto inner_dict = new_dict();
      for (const auto& [key, value] : mcclDump) {
        inner_dict.insert(key, value);
      }
      per_comm_dict.insert(mcclId, inner_dict);
    }
  }
  if (per_comm_dict.size() > 0) {
    result.insert(mccl_comm_key, per_comm_dict);
  }
  return pickle_str(result);
}

std::vector<MCCLTraceBuffer::Entry> MCCLTraceBuffer::dump_entries() {
  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<Entry> result;
  result.reserve(entries_.size());
  result.insert(result.end(), entries_.begin() + next_, entries_.end());
  result.insert(result.end(), entries_.begin(), entries_.begin() + next_);
  // query any remaining events
  for (auto& r : result) {
    update_state(r);
    r.start_ = r.end_ = nullptr;
  }
  return result;
}

void MCCLTraceBuffer::update_state(Entry& r) {
  if (r.start_ != nullptr) {
    bool started = r.start_->query();
    if (started && !r.time_discovered_started_) {
      r.time_discovered_started_ = c10::getTime();
    }
  }
  if (r.end_ != nullptr) {
    bool completed = r.end_->query();
    if (completed && !r.time_discovered_completed_) {
      r.time_discovered_completed_ = c10::getTime();
    }
  }
}

// from FlightRecorder.cpp

bool recursive_mkdir(const std::string& dir) {
  // Check if current dir exists
  const char* p_dir = dir.c_str();
  const bool dir_exists = (access(p_dir, F_OK) == 0);
  if (dir_exists) {
    return true;
  }

  // Find folder separator and check if we are at the top
  auto pos = dir.find_last_of("/\\");
  if (pos == std::string::npos) {
    return false;
  }

  // Try to create parent directory
  if (!(recursive_mkdir(dir.substr(0, pos)))) {
    return false;
  }

  // Try to create current directory
#ifdef _WIN32
  int ret = _mkdir(dir.c_str());
#else
  int ret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG);
#endif
  // Success
  if (ret == 0) {
    return true;
  }

  // Try to create complete path again
#ifdef _WIN32
  ret = _mkdir(dir.c_str());
#else
  ret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG);
#endif
  return ret == 0;
}

void DebugInfoWriter::write(const std::string& trace) {
  // Open a file for writing. The ios::binary flag is used to write data as
  // binary.
  std::ofstream file(filename_, std::ios::binary);

  // Check if the file was opened successfully.
  if (!file.is_open()) {
    LOG(ERROR) << "Error opening file for writing Flight Recorder debug info: "
               << filename_;
    return;
  }

  if (!file.write(trace.data(), static_cast<std::streamsize>(trace.size()))) {
    const auto bad = file.bad();
    LOG(ERROR) << "Error writing Flight Recorder debug info to file: "
               << filename_ << " bad bit: " << bad;
    return;
  }

  // Flush the buffer to ensure data is written to the file
  file.flush();
  if (file.bad()) {
    LOG(ERROR) << "Error flushing Flight Recorder debug info: " << filename_;
    return;
  }

  LOG(INFO) << "Finished writing Flight Recorder debug info to " << filename_;
}

DebugInfoWriter& DebugInfoWriter::getWriter(int rank) {
  if (writer_ == nullptr) {
    // Attempt to write to running user's HOME directory cache folder - if it
    // exists.
    auto homeDir = getCvarString({"HOME"}, "/tmp");
    std::string cacheDirPath = homeDir + "/.cache/torch";
    // Create the .cache directory if it doesn't exist
    recursive_mkdir(cacheDirPath);
    std::string defaultLocation = cacheDirPath + "/" + "mccl_trace_rank_";

    std::string fileNamePrefix = getCvarString(
        {"TORCH_MCCL_DEBUG_INFO_TEMP_FILE"}, defaultLocation.c_str());
    // Using std::unique_ptr here to auto-delete the writer object
    // when the pointer itself is destroyed.
    std::unique_ptr<DebugInfoWriter> writerPtr(
        new DebugInfoWriter(fileNamePrefix, rank));
    DebugInfoWriter::registerWriter(std::move(writerPtr));
  }
  return *writer_;
}

void DebugInfoWriter::registerWriter(std::unique_ptr<DebugInfoWriter> writer) {
  TORCH_CHECK_WITH(
      DistBackendError,
      hasWriterRegistered_.load() == false,
      "debugInfoWriter already registered");
  hasWriterRegistered_.store(true);
  writer_ = std::move(writer);
}

// Returns the traceback of current entry, in string form.
// Note: `getTraceback` invokes `torch::symbolize`, which may need to acquire
// the GIL. If you don't want to block the current thread or take the risk of a
// GIL deadlock, you can use an asynchronous calling mechanism like std::async.
std::string FlightRecorder::Entry::getTraceback() {
  torch::CapturedTraceback* traceback = traceback_.get();
  torch::SymbolizedTracebacks s_tbs = torch::symbolize({traceback});
  // We use 0 because we only have one traceback here.
  const auto& s_tb = s_tbs.tracebacks.at(0);
  std::stringstream oss;
  for (auto idx : c10::irange(s_tb.size())) {
    auto frame_id = s_tb[idx];
    const auto& frame = s_tbs.all_frames.at(frame_id);
    oss << "#" << idx << " " << frame.funcname << " from " << frame.filename
        << ":" << frame.lineno << '\n';
  }
  /* Resulted format is like:
    #0 all_reduce from pytorch/torch/distributed/distributed_c10d.py:2696
    #1 wrapper from pytorch/torch/distributed/c10d_logger.py:83
    #2 bar from /home/user/repro.py:15
    #3 foo from /home/user/repro.py:24
    #4 main from /home/user/repro.py:34
    #5 <module> from /home/user/repro.py:40
  */
  return oss.str();
}

std::optional<size_t> FlightRecorder::record(
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
    bool isP2P) {
  if (!enabled_) {
    return std::nullopt;
  }
  if (all_pg_status_.find(pg_id) == all_pg_status_.end()) {
    // Current pg_status is not in FR.
    all_pg_status_[pg_id] = std::move(pg_status);
  }
  auto traceback =
      torch::CapturedTraceback::gather(true, true, capture_cpp_stack_);
  std::lock_guard<std::mutex> guard(mutex_);

  auto te = Entry{
      id_,
      pg_id,
      pg_name,
      collective_seq_id,
      p2p_seq_id,
      op_id,
      std::move(profiling_name),
      std::move(traceback),
      start,
      end,
      c10::getTime(),
      timeout_ms.count(),
      isP2P,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      {},
      {},
      {},
      {},
      {},
      false};

  for (const auto& input : inputs) {
    c10::IntArrayRef sizes = input.sizes();
    te.input_dtypes_.push_back(input.dtype().toScalarType());
    te.input_dims_.push_back(static_cast<int64_t>(sizes.size()));
    te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
  }

  for (const auto& output : outputs) {
    c10::IntArrayRef sizes = output.sizes();
    te.output_dtypes_.push_back(output.dtype().toScalarType());
    te.output_dims_.push_back(static_cast<int64_t>(sizes.size()));
    te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
  }

  if (entries_.size() < max_entries_) {
    entries_.emplace_back(std::move(te));
  } else {
    entries_[next_++] = std::move(te);
    if (next_ == max_entries_) {
      next_ = 0;
    }
  }
  return id_++;
}

void FlightRecorder::record_pg_ranks(
    const std::tuple<std::string, std::string>& pg_name,
    std::vector<uint64_t> ranks) {
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> guard(mutex_);
  pg_name_to_ranks_[pg_name] = std::move(ranks);
}

void FlightRecorder::update_state(Entry& r) {
  if (r.start_ != nullptr) {
    bool started = r.start_->query();
    if (started && !r.time_discovered_started_) {
      r.time_discovered_started_ = c10::getTime();
    }
  }
  if (r.end_ != nullptr) {
    bool completed = r.end_->query();
    if (completed && !r.time_discovered_completed_) {
      r.time_discovered_completed_ = c10::getTime();
    }
  }
}

std::vector<FlightRecorder::Entry> FlightRecorder::dump_entries() {
  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<Entry> result;
  result.reserve(entries_.size());
  result.insert(
      result.end(),
      entries_.begin() + static_cast<std::ptrdiff_t>(next_),
      entries_.end());
  result.insert(
      result.end(),
      entries_.begin(),
      entries_.begin() + static_cast<std::ptrdiff_t>(next_));
  // query any remaining events
  for (auto& r : result) {
    update_state(r);
    r.start_ = r.end_ = nullptr;
  }
  return result;
}

// Returns the entry with the given id, if it exists. Otherwise, returns
// std::nullopt.
std::optional<FlightRecorder::Entry> FlightRecorder::getEntry(
    std::optional<size_t> id) {
  if (!enabled_ || !id) {
    return std::nullopt;
  }

  std::unique_lock<std::mutex> guard(mutex_);
  Entry entry = entries_.at(*id % max_entries_);
  if (entry.id_ == *id) {
    return entry;
  } else {
    return std::nullopt;
  }
}

void FlightRecorder::retire_id(
    std::optional<size_t> id,
    bool compute_duration) {
  if (!enabled_ || !id) {
    return;
  }

  bool can_compute_duration = false;
  Event* startEvent = nullptr;
  Event* endEvent = nullptr;
  std::optional<float> duration = std::nullopt;

  std::unique_lock<std::mutex> guard(mutex_);

  Entry* entry = &entries_.at(*id % max_entries_);
  if (entry->id_ == *id) {
    update_state(*entry);

    if (compute_duration) {
      can_compute_duration = entry->time_discovered_completed_.has_value() &&
          entry->start_ && entry->end_;
      startEvent = entry->start_;
      endEvent = entry->end_;
    }
    entry->retired_ = true;
    entry->start_ = entry->end_ = nullptr;
  }

  if (can_compute_duration) {
    // Compute duration without without holding the lock, because
    // musaEventDuration() can hang, and we need to acquire the lock before we
    // can dump(), which we never want to block.
    guard.unlock();
    duration = getDurationFromEvent(*startEvent, *endEvent);
    guard.lock();

    // Refresh the entry pointer, see if the entry has been overwritten
    entry = &entries_.at(*id % max_entries_);
    if (entry->id_ != *id) {
      LOG(INFO) << "retire_id abandoned for id " << *id
                << ", event was overwritten while waiting to compute duration.";
      return;
    }
    if (duration.has_value()) {
      entry->duration_ = duration;
    }
  }
}

const c10::List<c10::IValue> FlightRecorder::getCollectiveTrace(
    bool includeStacktraces,
    bool onlyActive) {
  auto entries = new_list();
  // Entries are returned in the order they were recorded
  auto result = dump_entries();
  std::vector<torch::CapturedTraceback*> tracebacks;
  torch::SymbolizedTracebacks stracebacks;
  std::vector<c10::IValue> all_frames;
  if (includeStacktraces) {
    for (auto& e : result) {
      tracebacks.push_back(e.traceback_.get());
    }
    stracebacks = torch::symbolize(tracebacks);
    for (const auto& f : stracebacks.all_frames) {
      auto d = new_dict();
      d.insert(name_key, f.funcname);
      d.insert(filename_key, f.filename);
      d.insert(line_key, int64_t(f.lineno));
      all_frames.emplace_back(std::move(d));
    }
  }
  for (auto i : c10::irange(result.size())) {
    auto dict = new_dict();
    auto& e = result.at(i);
    // Skip completed events
    if (onlyActive && e.time_discovered_completed_.has_value()) {
      continue;
    }
    if (includeStacktraces) {
      auto& tb = stracebacks.tracebacks.at(i);
      auto frames = new_list();
      for (auto frame : tb) {
        frames.push_back(all_frames.at(frame));
      }
      dict.insert(frames_key, frames);
    }

    dict.insert(record_id_key, int64_t(e.id_));
    dict.insert(pg_id_key, int64_t(e.pg_id_));
    dict.insert(pg_name_key, e.pg_name_);
    dict.insert(collective_seq_id_key, int64_t(e.collective_seq_id_));
    dict.insert(p2p_seq_id_key, int64_t(e.p2p_seq_id_));
    dict.insert(op_id_key, int64_t(e.op_id_));
    dict.insert(profiling_name_key, e.profiling_name_);
    dict.insert(time_created_key, int64_t(e.time_created_));
    if (e.duration_) {
      dict.insert(duration_key, *e.duration_);
    }

    auto it = e.sizes_.begin();
    auto read_sizes = [&](const c10::SmallVector<int64_t, 4>& dims) {
      auto sizes = new_list();
      for (auto dim : dims) {
        auto arg_sizes = new_list();
        for ([[maybe_unused]] auto i : c10::irange(dim)) {
          arg_sizes.push_back(*it++);
        }
        sizes.push_back(arg_sizes);
      }
      return sizes;
    };

    dict.insert(input_sizes_key, read_sizes(e.input_dims_));
    std::vector<std::string> input_dtypes_strs;
    input_dtypes_strs.reserve(e.input_dtypes_.size());
    for (const auto& input_dtype : e.input_dtypes_) {
      input_dtypes_strs.emplace_back(c10::toString(input_dtype));
    }
    dict.insert(input_dtypes_key, input_dtypes_strs);
    dict.insert(output_sizes_key, read_sizes(e.output_dims_));
    std::vector<std::string> output_dtypes_strs;
    output_dtypes_strs.reserve(e.output_dtypes_.size());
    for (const auto& output_dtype : e.output_dtypes_) {
      output_dtypes_strs.emplace_back(c10::toString(output_dtype));
    }
    dict.insert(output_dtypes_key, output_dtypes_strs);
    if (e.time_discovered_completed_.has_value()) {
      dict.insert(state_key, completed_state);
    } else if (e.time_discovered_started_.has_value()) {
      dict.insert(state_key, started_state);
    } else {
      dict.insert(state_key, scheduled_state);
    }

    dict.insert(
        time_discovered_started_key,
        e.time_discovered_started_.has_value()
            ? int64_t(*e.time_discovered_started_)
            : c10::IValue());
    dict.insert(
        time_discovered_completed_key,
        e.time_discovered_completed_.has_value()
            ? int64_t(*e.time_discovered_completed_)
            : c10::IValue());
    dict.insert(retired_key, e.retired_);
    dict.insert(timeout_key, e.timeout_ms_);
    dict.insert(is_p2p_key, e.isP2P_);

    entries.push_back(dict);
  }
  return entries;
}

const c10::Dict<c10::IValue, c10::IValue> FlightRecorder::getPgConfig() {
  auto pg_config = new_dict();
  for (const auto& [pg_name, ranks] : pg_name_to_ranks_) {
    auto pg_info = new_dict();
    pg_info.insert("name", std::get<0>(pg_name));
    pg_info.insert("desc", std::get<1>(pg_name));
    pg_info.insert("ranks", ranks_str(ranks));
    pg_config.insert(std::get<0>(pg_name), pg_info);
  }
  return pg_config;
}

const std::map<std::string, std::map<std::string, std::string>> FlightRecorder::
    getPgConfigJson() {
  std::map<std::string, std::map<std::string, std::string>> result;
  for (const auto& [pg_name, ranks] : pg_name_to_ranks_) {
    auto pg_info = std::map<std::string, std::string>();
    pg_info["name"] = std::get<0>(pg_name);
    pg_info["desc"] = std::get<1>(pg_name);
    pg_info["ranks"] = ranks_str(ranks);
    result.emplace(std::get<0>(pg_name), pg_info);
  }
  return result;
}

const c10::Dict<c10::IValue, c10::IValue> FlightRecorder::getPgStatus() {
  auto all_pg_status = new_dict();
  for (const auto& [pg_id, status] : all_pg_status_) {
    auto pg_status = new_dict();
    pg_status.insert("last_enqueued_collective", status->lastEnqueuedSeq);
    pg_status.insert("last_started_collective", status->lastStartedSeq);
    pg_status.insert("last_completed_collective", status->lastCompletedSeq);
    all_pg_status.insert(std::to_string(pg_id), pg_status);
  }
  return all_pg_status;
}

const std::map<std::string, std::map<std::string, std::string>> FlightRecorder::
    getPgStatusJson() {
  std::map<std::string, std::map<std::string, std::string>> result;
  for (const auto& [pg_id, status] : all_pg_status_) {
    auto pg_status = std::map<std::string, std::string>();
    pg_status["last_enqueued_collective"] =
        std::to_string(status->lastEnqueuedSeq);
    pg_status["last_started_collective"] =
        std::to_string(status->lastStartedSeq);
    pg_status["last_completed_collective"] =
        std::to_string(status->lastCompletedSeq);
    result[std::to_string(pg_id)] = pg_status;
  }
  return result;
}

// std::string FlightRecorder::dump_json(
//     const std::optional<std::unordered_map<
//         std::string,
//         std::unordered_map<std::string, std::string>>>& mcclDumpMap,
//     bool includeCollectives,
//     bool onlyActive) {
// }

std::string FlightRecorder::dump(
    const std::optional<std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::string>>>& mcclDumpMap,
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  // STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.FlightRecorder__dump);
  auto result = new_dict();
  // common values
  result.insert(version_key, version_val);
  result.insert(pg_config_key, getPgConfig());
  result.insert(pg_status_key, getPgStatus());

  // collective trace
  if (includeCollectives) {
    result.insert(
        entries_key, getCollectiveTrace(includeStackTraces, onlyActive));
  }
  // convert mcclDumpMap into a dictionary
  auto per_comm_dict = new_dict();
  if (mcclDumpMap.has_value()) {
    for (const auto& [mcclId, mcclDump] : mcclDumpMap.value()) {
      auto inner_dict = new_dict();
      for (const auto& [key, value] : mcclDump) {
        inner_dict.insert(key, value);
      }
      per_comm_dict.insert(mcclId, inner_dict);
    }
  }
  if (!per_comm_dict.empty()) {
    result.insert(mccl_comm_key, per_comm_dict);
  }
  return pickle_str(result);
}

std::unique_ptr<DebugInfoWriter> DebugInfoWriter::writer_ = nullptr;
std::atomic<bool> DebugInfoWriter::hasWriterRegistered_(false);

float getDurationFromEvent(
    at::musa::MUSAEvent& mcclStartEvent,
    at::musa::MUSAEvent& mcclEndEvent) {
  TORCH_CHECK(
      mcclEndEvent.query(),
      "getDuration can only be called after work is succeeded.")
  return mcclStartEvent.elapsed_time(mcclEndEvent);
}

} // namespace c10d

// mcclUtils.cpp has nothing to do.