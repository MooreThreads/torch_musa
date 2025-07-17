#include "torch_musa/csrc/distributed/MCCLUtils.h"
#include <c10/util/CallOnce.h>

#include <mutex>
namespace c10d {
mcclComm_t MCCLComm::getMcclComm() {
  std::unique_lock<std::mutex> lock(mutex_);
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
  return mcclComm_;
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
        std::unordered_map<std::string, std::string>>>& ncclDumpMap,
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
  if (ncclDumpMap.has_value()) {
    for (const auto& [ncclId, ncclDump] : ncclDumpMap.value()) {
      auto inner_dict = new_dict();
      for (const auto& [key, value] : ncclDump) {
        inner_dict.insert(key, value);
      }
      per_comm_dict.insert(ncclId, inner_dict);
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

std::unique_ptr<DebugInfoWriter> DebugInfoWriter::writer_ = nullptr;
std::atomic<bool> DebugInfoWriter::hasWriterRegistered_(false);

void DebugInfoWriter::write(const std::string& ncclTrace) {
  // Open a file for writing. The ios::binary flag is used to write data as
  // binary.
  std::ofstream file(filename_, std::ios::binary);

  // Check if the file was opened successfully.
  if (!file.is_open()) {
    LOG(ERROR) << "Error opening file for writing NCCLPG debug info: "
               << filename_;
    return;
  }

  file.write(ncclTrace.data(), ncclTrace.size());
  LOG(INFO) << "Finished writing NCCLPG debug info to " << filename_;
}

DebugInfoWriter& DebugInfoWriter::getWriter(int rank) {
  if (writer_ == nullptr) {
    std::string fileNamePrefix = getCvarString(
        {"TORCH_NCCL_DEBUG_INFO_TEMP_FILE"}, "/tmp/nccl_trace_rank_");
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

} // namespace c10d

// mcclUtils.cpp has nothing to do.