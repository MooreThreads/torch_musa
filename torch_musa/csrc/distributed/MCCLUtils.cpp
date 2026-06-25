#include "torch_musa/csrc/distributed/MCCLUtils.h"
#include <c10/util/CallOnce.h>
#include <c10/util/env.h>
#include <c10/util/hash.h>

#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

#include "torch_musa/csrc/core/MUSAGuard.h"

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
  std::swap(uniqueHash_, other.uniqueHash_);
  std::swap(aborted_, other.aborted_);
  std::swap(mcclAsyncErr_, other.mcclAsyncErr_);
  std::swap(initialized_, other.initialized_);
  std::swap(nonBlocking_, other.nonBlocking_);
  std::swap(deviceIndex_, other.deviceIndex_);
}

void MCCLComm::setUniqueHash(mcclUniqueId mcclId) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&mcclId);

  std::ostringstream oss;
  for (int i = 0; i < MCCL_UNIQUE_ID_BYTES; ++i) {
    oss << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<int>(bytes[i]);
  }
  uniqueHash_ = oss.str();
}

void MCCLComm::setUniqueHash(std::string hash) {
  uniqueHash_ = std::move(hash);
}

std::string MCCLComm::getUniqueHash() {
  return uniqueHash_;
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
  comm->setUniqueHash(commId);
  comm->rank_ = rank;
  comm->deviceIndex_ = deviceIndex;
  comm->initialized_ = true;
  // Old style comm is always blocking.
  comm->nonBlocking_ = false;
  return comm;
}

#ifdef MCCL_HAS_CONFIG
std::shared_ptr<MCCLComm> MCCLComm::create(
    int numRanks,
    int rank,
    mcclUniqueId commId,
    at::DeviceIndex deviceIndex,
    mcclConfig_t& config) {
  at::musa::OptionalMUSAGuard gpuGuard(deviceIndex);
  auto comm = std::make_shared<MCCLComm>();
  comm->nonBlocking_ = config.blocking == 0;
  LOG(INFO) << "Rank " << rank << ": creating MCCL communicator with mode: "
            << (comm->nonBlocking_ ? "nonblocking" : "blocking");
  C10D_MCCL_CHECK_NONBLOCKING(
      mcclCommInitRankConfig(
          &(comm->mcclComm_), numRanks, commId, rank, &config),
      std::nullopt);
  comm->setUniqueHash(commId);
  comm->rank_ = rank;
  comm->deviceIndex_ = deviceIndex;
  comm->initialized_ = !comm->nonBlocking_;
  return comm;
}
#endif

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
#ifdef MCCL_HAS_COMM_NONBLOCKING
  // In non-blocking mode, ensure comm is ready.
  if (nonBlocking_) {
    // Wait with long interval if communicator is being initialized.
    bool longInterval = !initialized_;
    waitReady(longInterval);
    // mcclComm_ should be initialized by now.
  }
#endif // MCCL_HAS_COMM_NONBLOCKING
  if (!initialized_) {
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
#ifdef MCCL_HAS_COMM_NONBLOCKING
  if (nonBlocking_) {
    // If timeout is reached, throw an exception.
    if (longInterval) {
      C10D_MCCL_CHECK_TIMEOUT_SLEEP(
          mcclInProgress, mcclComm_, commFailureReason_);
    } else {
      C10D_MCCL_CHECK_TIMEOUT(mcclInProgress, mcclComm_, commFailureReason_);
    }
  }
#else
  (void)longInterval;
#endif // MCCL_HAS_COMM_NONBLOCKING
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
  mcclComm_ = nullptr;
}

void MCCLComm::abort(std::optional<std::string> commFailureReason) {
  LockType lock(mutex_);
  at::musa::OptionalMUSAGuard gpuGuard(deviceIndex_);
  if (aborted_ && !initialized_) {
    // Should not abort twice.
    return;
  }
#ifdef MCCL_HAS_COMM_REGISTER
  for (const auto& it : registeredSegmentHandles_) {
    void* handle = it.second;
    C10D_MCCL_CHECK(
        ::mcclCommDeregister(mcclComm_, handle),
        c10::str(
            "Failed to deregister segment handle ",
            handle,
            " on mcclComm_ ",
            mcclComm_));
  }
  registeredSegmentHandles_.clear();
#endif

  // Set true failure reason if provided by ProcessGroupMCCL (e.g. work
  // timeout)
  commFailureReason_ = commFailureReason;
  LOG(INFO) << "Aborting mcclComm_ " << mcclComm_ << " with reason: "
            << (commFailureReason ? *commFailureReason
                                  : "No abort reason provided.");

#ifndef MCCL_HAS_COMM_NONBLOCKING
  C10D_MCCL_CHECK(::mcclCommAbort(mcclComm_), commFailureReason_);
#else
  C10D_MCCL_CHECK_TIMEOUT(
      ::mcclCommAbort(mcclComm_), mcclComm_, commFailureReason_);
#endif

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
    bool errorOnRereg /*=true*/,
    bool window /*=false*/) {
#ifdef MCCL_HAS_COMM_REGISTER
  if (registeredSegmentHandles_.count(ptr) > 0) {
    TORCH_CHECK(
        !errorOnRereg,
        "Segment with ptr ",
        ptr,
        " has already been registered on mcclComm_ ",
        mcclComm_);
    return mcclSuccess;
  }

  void* handle = nullptr;
  // Use getMcclComm to make sure comm is ready before calling mccl APIs
  auto comm = getMcclComm();
#ifdef MCCL_HAS_COMM_WINDOW_REGISTER
  if (window) {
    C10D_MCCL_CHECK(
        mcclCommWindowRegister(
            comm,
            ptr,
            size,
            reinterpret_cast<mcclWindow_t*>(&handle),
            MCCL_WIN_COLL_SYMMETRIC),
        c10::str(
            "Failed to window register segment with ptr ",
            ptr,
            ", size ",
            size,
            " on mcclComm_ ",
            comm));
  } else {
    C10D_MCCL_CHECK(
        mcclCommRegister(comm, ptr, size, &handle),
        c10::str(
            "Failed to register segment with ptr ",
            ptr,
            ", size ",
            size,
            " on mcclComm_ ",
            comm));
  }
#else
  C10D_MCCL_CHECK(
      mcclCommRegister(comm, ptr, size, &handle),
      c10::str(
          "Failed to register segment with ptr ",
          ptr,
          ", size ",
          size,
          " on mcclComm_ ",
          comm));
#endif
  registeredSegmentHandles_[ptr] = handle;
  return mcclSuccess;
#else
  return mcclInvalidUsage;
#endif
}

mcclResult_t MCCLComm::deregisterSegment(void* ptr, bool window /*=false*/) {
  LockType lock(mutex_);
#ifdef MCCL_HAS_COMM_REGISTER
  TORCH_CHECK(
      registeredSegmentHandles_.count(ptr) == 1,
      "Segment with ptr ",
      ptr,
      " is not registered on mcclComm_ ",
      mcclComm_);
  void* handle = registeredSegmentHandles_[ptr];
  auto comm = getMcclComm();
#ifdef MCCL_HAS_COMM_WINDOW_REGISTER
  if (window) {
    C10D_MCCL_CHECK(
        mcclCommWindowDeregister(comm, reinterpret_cast<mcclWindow_t>(handle)),
        c10::str(
            "Failed to window deregister segment handle ",
            handle,
            ", with ptr ",
            ptr,
            " on mcclComm_ ",
            comm));
  } else {
    C10D_MCCL_CHECK(
        mcclCommDeregister(comm, handle),
        c10::str(
            "Failed to deregister segment handle ",
            handle,
            ", with ptr ",
            ptr,
            " on mcclComm_ ",
            comm));
  }
#else
  C10D_MCCL_CHECK(
      mcclCommDeregister(comm, handle),
      c10::str(
          "Failed to deregister segment handle ",
          handle,
          ", with ptr ",
          ptr,
          " on mcclComm_ ",
          comm));
#endif
  registeredSegmentHandles_.erase(ptr);
  return mcclSuccess;
#else
  return mcclInvalidUsage;
#endif
}

at::DeviceIndex MCCLComm::getDeviceIndex() {
  return deviceIndex_;
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

// Default value: 30 minutes.
int mccl_nonblocking_timeout() {
  static int timeout = -2; // -2 means not initialized.
  if (timeout == -2) {
    auto val = c10::utils::get_env("TORCH_MCCL_NONBLOCKING_TIMEOUT");
    if ((!val.has_value() || val->empty())) {
      // Backward-compatible alias shared with NCCL backend.
      val = c10::utils::get_env("TORCH_NCCL_NONBLOCKING_TIMEOUT");
    }
    if (val.has_value() && !val->empty()) {
      timeout = std::stoi(*val);
    } else {
      // Default value consistent with kBackendDefaultTimeout.
      timeout = 30 * 60;
    }
  }
  return timeout;
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

size_t hashTensors(const std::vector<at::Tensor>& tensors) {
  size_t hash = 0;
  for (const auto& tensor : tensors) {
    if (tensor.numel() > 0 && tensor.storage()) {
      size_t data_size = tensor.storage().nbytes();
      if (data_size > 0 && tensor.storage().data_ptr()) {
        auto src = static_cast<const char*>(tensor.storage().data_ptr().get());
        std::vector<char> dst(data_size);
        if (tensor.is_musa()) {
          // Trigger synchronization so collective outputs are materialized
          // before we hash them.
          C10_MUSA_CHECK(
              musaMemcpy(dst.data(), src, data_size, musaMemcpyDeviceToHost));
        } else {
          std::memcpy(dst.data(), src, data_size);
        }
        for (size_t i = 0; i < data_size; ++i) {
          hash = c10::hash_combine(hash, c10::get_hash(dst[i], data_size));
        }
      }
    }
  }
  return hash;
}

mcclDataType_t getMcclDataType(at::ScalarType type) {
  auto it = mcclDataType.find(type);
  TORCH_CHECK_WITH(
      TypeError,
      it != mcclDataType.end(),
      "Input tensor data type is not supported for MCCL process group: ",
      type);
  return it->second;
}

} // namespace c10d
