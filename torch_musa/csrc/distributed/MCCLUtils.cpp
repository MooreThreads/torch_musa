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
} // namespace c10d

// mcclUtils.cpp has nothing to do.