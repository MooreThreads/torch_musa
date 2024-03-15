#ifndef TORCH_MUSA_CSRC_DISTRIBUTED_MCCLUTILS_H_
#define TORCH_MUSA_CSRC_DISTRIBUTED_MCCLUTILS_H_

#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <mutex>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <mccl.h>

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

namespace c10d {

std::string getMcclVersion();
std::string mcclGetErrorWithVersion(mcclResult_t error);
std::string getMcclErrorDetailStr(
    mcclResult_t error,
    c10::optional<std::string> processGroupFailureReason = c10::nullopt);

// RAII wrapper for MCCL communicator
class MCCLComm {
 public:
  explicit MCCLComm(mcclComm_t mcclComm)
      : mcclComm_(mcclComm),
        aborted_(false),
        mcclAsyncErr_(mcclSuccess),
        commFailureReason_(c10::nullopt) {}

  MCCLComm() : MCCLComm(nullptr) {}

  ~MCCLComm() noexcept {
    // Add lock in this destructor, as aborted_ needs to be read after memory
    // barrier here.
    std::unique_lock<std::mutex> lock(mutex_);
    if (mcclComm_ && !aborted_) {
      C10D_MCCL_ASSERT(::mcclCommAbort(mcclComm_));
    }
  }

  static std::shared_ptr<MCCLComm> create(
      int numRanks,
      int rank,
      mcclUniqueId commId) {
    auto comm = std::make_shared<MCCLComm>();
    C10D_MCCL_CHECK(
        mcclCommInitRank(&(comm->mcclComm_), numRanks, commId, rank),
        c10::nullopt);
    comm->mcclId_ = commId;
    comm->rank_ = rank;
    return comm;
  }

  mcclUniqueId getMcclId() {
    return mcclId_;
  }

  // Must not be copyable
  MCCLComm(const MCCLComm&) = delete;
  MCCLComm& operator=(const MCCLComm&) = delete;

  // Do not support move assignment as there is no valid use case
  MCCLComm& operator=(MCCLComm&& other) = delete;

  // Move constructable
  MCCLComm(MCCLComm&& other) {
    // Using other's lock, as it reads other's states
    // Can not use this.mutex_, as this object is being constructed.
    std::unique_lock<std::mutex> lock(other.mutex_);
    std::swap(mcclComm_, other.mcclComm_);
    std::swap(aborted_, other.aborted_);
    std::swap(mcclAsyncErr_, other.mcclAsyncErr_);
  }

  mcclComm_t getMcclComm();

  c10::optional<std::string> getMcclCommFailureReason() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return commFailureReason_;
  }

  void mcclCommAbort(
      c10::optional<std::string> commFailureReason = c10::nullopt) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (aborted_) {
      // Should not abort twice.
      return;
    }

    // Set true failure reason if provided by ProcessGroupMCCL (e.g. work
    // timeout)
    commFailureReason_ = commFailureReason;

    C10D_MCCL_CHECK(::mcclCommAbort(mcclComm_), commFailureReason_);
    aborted_ = true;
    mcclComm_ = nullptr;

    // Set an appropriate error so that we avoid using the communicator.
    if (mcclAsyncErr_ == mcclSuccess) {
      mcclAsyncErr_ = mcclSystemError;
    }
  }

  bool isAborted() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return aborted_;
  }

  mcclResult_t checkForMcclError() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (mcclAsyncErr_ != mcclSuccess) {
      return mcclAsyncErr_;
    }
    C10D_MCCL_CHECK(
        mcclCommGetAsyncError(mcclComm_, &mcclAsyncErr_), commFailureReason_);
    return mcclAsyncErr_;
  }

 protected:
  mcclComm_t mcclComm_;
  // Unique mccl_id for this communicator.
  mcclUniqueId mcclId_;
  bool aborted_;
  mcclResult_t mcclAsyncErr_;
  mutable std::mutex mutex_;
  // Rank that this communicator corresponds to.
  int rank_;
  // Optional reason for communicator failure, provided by ProcessGroupMCCL for
  // better error messaging.
  c10::optional<std::string> commFailureReason_;
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

} // namespace c10d

#endif // TORCH_MUSA_CSRC_DISTRIBUTED_MCCLUTILS_H_