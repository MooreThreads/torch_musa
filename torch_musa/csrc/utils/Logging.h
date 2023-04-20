#ifndef TORCH_MUSA_CSRC_UTILS_LOGGING_H_
#define TORCH_MUSA_CSRC_UTILS_LOGGING_H_
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace torch_musa {
namespace logging {

// Define the severity for logging.
enum class Severity { kInfo = 0, kWarning = 1, kFatal = 2 };

template <typename X, typename Y>
std::unique_ptr<std::string> ToString(const X& x, const Y& y) {
  std::ostringstream os;
  os << " (" << x << " vs " << y << ") ";
  return std::unique_ptr<std::string>(new std::string(os.str()));
}

#define DEFINE_COMPARISON_CHECKER(name, op)                                 \
  template <typename L, typename R>                                         \
  inline std::unique_ptr<std::string> ToString##name(                       \
      const L& left, const R& right) {                                      \
    return (left op right) ? nullptr : ToString(left, right);               \
  }                                                                         \
  inline std::unique_ptr<std::string> ToString##name(int left, int right) { \
    return ToString##name<int, int>(left, right);                           \
  }

class LogMessage {
 public:
  LogMessage(const char* file, int line, Severity severity)
      : severity_(severity) {
    stream_ << "[" << TimeToString() << "] " << file << ":" << line << ": ";
  }

  ~LogMessage() {
    stream_ << '\n';
    std::cerr << stream_.str();
    if (severity_ != Severity::kInfo) {
      stream_.flush();
    }
    if (severity_ == Severity::kFatal) {
      Fatal();
    }
  }

  // Directly aborts.
  void Fatal() {
    abort();
  }

  // Return the stream associated to the logger.
  std::stringstream& stream() {
    return stream_;
  }

 private:
  std::string TimeToString() const {
    // TODO(mt-ai) Handle timer for different OS.
    auto time_value = std::time(nullptr);
    auto now = *std::localtime(&time_value);

    std::ostringstream oss;
    oss << std::put_time(&now, "%H-%M-%S");
    return oss.str();
  }

  Severity severity_;
  std::stringstream stream_;
};

// This class is used to explicitly ignore values in the conditional
// logging macros. This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LogMessageVoidify {
 public:
  LogMessageVoidify() = default;
  // This has to be an operator with a precedence lower than << but
  // higher than "?:".
  void operator&(std::ostream&) {}
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
DEFINE_COMPARISON_CHECKER(LT, <)
DEFINE_COMPARISON_CHECKER(GT, >)
DEFINE_COMPARISON_CHECKER(LE, <=)
DEFINE_COMPARISON_CHECKER(GE, >=)
DEFINE_COMPARISON_CHECKER(EQ, ==)
DEFINE_COMPARISON_CHECKER(NE, !=)
#pragma GCC diagnostic pop

#define CHECK_BINARY_OP(name, op, x, y)                          \
  if (auto msg = torch_musa::logging::ToString##name(x, y))      \
  torch_musa::logging::LogMessage(                               \
      __FILE__, __LINE__, torch_musa::logging::Severity::kFatal) \
          .stream()                                              \
      << "Check failed: " << #x " " #op " " #y << *msg << ": "

#define TORCH_MUSA_CHECK_LT(x, y) CHECK_BINARY_OP(LT, <, x, y)
#define TORCH_MUSA_CHECK_GT(x, y) CHECK_BINARY_OP(GT, >, x, y)
#define TORCH_MUSA_CHECK_LE(x, y) CHECK_BINARY_OP(LE, <=, x, y)
#define TORCH_MUSA_CHECK_GE(x, y) CHECK_BINARY_OP(GE, >=, x, y)
#define TORCH_MUSA_CHECK_EQ(x, y) CHECK_BINARY_OP(EQ, ==, x, y)
#define TORCH_MUSA_CHECK_NE(x, y) CHECK_BINARY_OP(NE, !=, x, y)

#ifndef NDEBUG
#define TORCH_MUSA_DCHECK_EQ(val1, val2) DCHECK_EQ(val1, val2)
#define TORCH_MUSA_DCHECK_NE(val1, val2) DCHECK_NE(val1, val2)
#define TORCH_MUSA_DCHECK_LE(val1, val2) DCHECK_LE(val1, val2)
#define TORCH_MUSA_DCHECK_LT(val1, val2) DCHECK_LT(val1, val2)
#define TORCH_MUSA_DCHECK_GE(val1, val2) DCHECK_GE(val1, val2)
#define TORCH_MUSA_DCHECK_GT(val1, val2) DCHECK_GT(val1, val2)
#else // !NDEBUG
#define TORCH_MUSA_DCHECK_EQ(val1, val2) \
  while (false)                          \
  DCHECK_EQ(val1, val2)
#define TORCH_MUSA_DCHECK_NE(val1, val2) \
  while (false)                          \
  DCHECK_NE(val1, val2)
#define TORCH_MUSA_DCHECK_LE(val1, val2) \
  while (false)                          \
  DCHECK_LE(val1, val2)
#define TORCH_MUSA_DCHECK_LT(val1, val2) \
  while (false)                          \
  DCHECK_LT(val1, val2)
#define TORCH_MUSA_DCHECK_GE(val1, val2) \
  while (false)                          \
  DCHECK_GE(val1, val2)
#define TORCH_MUSA_DCHECK_GT(val1, val2) \
  while (false)                          \
  DCHECK_GT(val1, val2)
#endif // NDEBUG

#define LOG_INFO                                                \
  torch_musa::logging::LogMessage(                              \
      __FILE__, __LINE__, torch_musa::logging::Severity::kInfo) \
      .stream()
#define LOG_WARNING                                                \
  torch_musa::logging::LogMessage(                                 \
      __FILE__, __LINE__, torch_musa::logging::Severity::kWarning) \
      .stream()
#define LOG_FATAL                                                \
  torch_musa::logging::LogMessage(                               \
      __FILE__, __LINE__, torch_musa::logging::Severity::kFatal) \
      .stream()

#ifndef NDEBUG
#define DLOG_INFO                                               \
  torch_musa::logging::LogMessage(                              \
      __FILE__, __LINE__, torch_musa::logging::Severity::kInfo) \
      .stream()
#define DLOG_WARNING                                               \
  torch_musa::logging::LogMessage(                                 \
      __FILE__, __LINE__, torch_musa::logging::Severity::kWarning) \
      .stream()
#define DLOG_FATAL                                               \
  torch_musa::logging::LogMessage(                               \
      __FILE__, __LINE__, torch_musa::logging::Severity::kFatal) \
      .stream()
#else // !NDEBUG
#define DLOG_INFO \
  while (false)   \
  LOG_INFO
#define DLOG_WARNING \
  while (false)      \
  LOG_WARNING
#define DLOG_FATAL \
  while (false)    \
  LOG_FATAL
#endif // NDEBUG
} // namespace logging
} // namespace torch_musa

#endif // TORCH_MUSA_CSRC_UTILS_LOGGING_H_
