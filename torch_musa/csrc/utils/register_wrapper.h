#ifndef TORCH_MUSA_CSRC_UTILS_REGISTER_WRAPPER_H
#define TORCH_MUSA_CSRC_UTILS_REGISTER_WRAPPER_H
/*
A wrapper of TORCH_LIBRARY_IMPL, with templated wrapper function.
Use config files or Environment to Enable/Disable print of operator info.
*/

#include <ATen/core/Formatting.h>
#include <c10/core/QScheme.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <sys/stat.h>
#include <torch/library.h>
#include <unistd.h>
#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <sstream>

/*
    To support enable/disable Wrapper of a specified function,
    we have to give the wrapper a function name(the func).
    However, C++ template before C++20 cannot support a char* or
    string inputs. Thus we have to wrap the full template into a
    macro.

Usage:

  From:
    ... func(...);
    ...
    TORCH_LIBRARY_IMPL(aten, PrivatedUse1, m) {
        m.impl("yaml", func);
    }

  To:

    1. Normal case:
    at::Tensor Abs(...);
    ...
    ADVANCED_REGISTER(aten, PrivateUse1, "abs", Abs);

    2. Not name function case:
    using at::native::unfold;
    ...
    REGISTER_IMPL(aten, PrivateUse1, "unfold", at::native::unfold,
at_native_unfold)

    3. Alias case:
    at::Tensor NotEqualTensor(...);
    ...
    ADVANCED_REGISTER(aten, PrivateUse1, "ne.Tensor", NotEqualTensor)
    REDEFINE_REGISTER(aten, PrivateUse1, "not_equal.Tensor", NotEqualTensor)

    4. Not name function alias case:
    using at::native::sth;
    ...
    REGISTER_IMPL(aten, PrivateUse1, "sth", at::native::sth, at_native_sth)
    REDEFINE_REGISTER(aten, PrivateUse1, "sth_alias", at_native_sth)
*/

/*
REGISTER_IMPL will register a wrapper named as wrapper_{name}.
*/

#define REGISTER_IMPL(lib, key, yaml, func, name)               \
  using namespace at::musa;                                     \
  template <class F, F f>                                       \
  struct wrapper_##name;                                        \
  template <class R, class... Args, R (*f)(Args...)>            \
  struct wrapper_##name<R (*)(Args...), f> {                    \
    static R wrap(Args... args) {                               \
      if (!GlobalConfig.IsOpEnabled(yaml, #func)) {             \
        return f(args...);                                      \
      }                                                         \
      GlobalConfig.CreateKernelStream(yaml, #func);             \
      GlobalConfig.enabled_ = false;                            \
      TraversalItems(args...);                                  \
      GlobalConfig.enabled_ = true;                             \
      GlobalConfig.SplitKernelIO();                             \
      R&& result = f(args...);                                  \
      GlobalConfig.enabled_ = false;                            \
      TraversalItems(result);                                   \
      GlobalConfig.enabled_ = true;                             \
      GlobalConfig.CloseKernelStream();                         \
      return std::forward<R>(result);                           \
    }                                                           \
  };                                                            \
  template <class... Args, void (*f)(Args...)>                  \
  struct wrapper_##name<void (*)(Args...), f> {                 \
    static void wrap(Args... args) {                            \
      if (!GlobalConfig.IsOpEnabled(yaml, #func)) {             \
        return f(args...);                                      \
      }                                                         \
      GlobalConfig.CreateKernelStream(yaml, #func);             \
      GlobalConfig.enabled_ = false;                            \
      TraversalItems(args...);                                  \
      GlobalConfig.enabled_ = true;                             \
      GlobalConfig.SplitKernelIO();                             \
      f(args...);                                               \
      GlobalConfig.CloseKernelStream();                         \
    }                                                           \
  };                                                            \
  TORCH_LIBRARY_IMPL(lib, key, m) {                             \
    m.impl(yaml, &wrapper_##name<decltype(&func), func>::wrap); \
  }

// lib = aten, key = PrivateUse1, yaml = torch op yaml name, func = kernel.
#define ADVANCED_REGISTER(lib, key, yaml, func) \
  REGISTER_IMPL(lib, key, yaml, func, func)

/*
As REGISTER_IMPL will generate a wrapper function. It will cause
redefinition of 'struct at::musa::wrapper_xxx<...' error, and we can
use REDEFINE_REGISTER to avoid the error.

It uses like:
ADVANCED_REGISTER(lib, key, "a.out", afunc)
REDEFINE_REGISTER(lib, key, "a.alias.out", afunc)

or
REGISTER_IMPL(lib, key, "b.out", at::musa::bfunc, at_musa_bfunc)
REDEFINE_REGISTER(lib, key, "b.alias.out", at_musa_bfunc)
*/

#define REDEFINE_REGISTER(lib, key, yaml, func)                 \
  TORCH_LIBRARY_IMPL(lib, key, m) {                             \
    m.impl(yaml, &wrapper_##func<decltype(&func), func>::wrap); \
  }
// End of Advance Register.

namespace at {
namespace musa {

class Config {
 public:
  Config(); // done.
  bool IsOpEnabled(const char* yaml, const char* func); // done.
  void set_enabled(bool flag);
  void set_dir(const std::string& dir);
  void CreateKernelStream(
      const char* yaml,
      const char* name); // call this before process input.
  void SplitKernelIO(); // call this between input and output.
  void CloseKernelStream(); // call this after a kernel finished.
  void TensorProcessor(const Tensor& t);
  std::string arg_log_name; // log_name is a string, open it every time to avoid
                            // IO error.
  std::string
      base_log; // base_log is for full model, at base_dir_/full_log.txt.
  std::string tensor_file_name;
  bool enabled_ = true;

 private:
  void InitConfig(); // Disabled.
  void LoadEnv(); // done.
  int level_; // 1 .. 6;
  int tensor_max_size_;
  bool has_op_white_list_ = false;
  bool has_op_black_list_ = false;
  void CreateTensorStream(); // generate a new file for next tensor.
  void CloseTensorStream(); // close file of this tensor.
  void TryCreateBaseDir();

  std::vector<std::string> op_white_list_;
  std::vector<std::string> op_black_list_;
  /*
  save_dir_ = base_dir_/[split_dirs_[0..N]]
  split_dirs_[X] = dir_numbers_[X] + last_dir_(at layer X);
  last_dir_ = op_name;
  dir_numbers_[X] = index of last
  */
  bool base_dir_created_ = false;
  std::string save_dir_;
  std::string base_dir_;
  std::string last_dir_;
  std::vector<std::string> split_dirs_;
  std::vector<int> dir_numbers_;
  bool is_output_ = false;
};

// Only init once in reigster_wrapper.cpp
extern Config GlobalConfig;

struct TensorSignature {
  bool is_bool_tensor;
  bool has_num; // if tensor not full of nan&inf
  int true_num; // only for bool tensor.
  bool has_nan;
  bool has_inf;
  int tensor_size;
  float max_value, min_value, avg;
  float nearest_zero_pos, nearest_zero_neg;
  float sum, abs_sum, standard_deviation; // Skip inf and nan.
  friend std::ostream& operator<<(
      std::ostream& ostr,
      const TensorSignature& ts);
};

/*
Helper Templates for iterators.
*/

template <typename, typename = void>
constexpr bool is_iterable{false};

template <typename T>
constexpr bool is_iterable<
    T,
    std::void_t<
        decltype(std::declval<T>().begin()),
        decltype(std::declval<T>().end())>> = true;

/*
General Templates.

Attension: only inline template functions can be defined in head file.

Attension: Primary Template must be put before specified.
*/

template <typename T>
inline void ProcessArgs(T& item) {
  std::ofstream arg_stream(GlobalConfig.arg_log_name, std::ios::app);
  if constexpr (is_iterable<T>) {
    arg_stream << "Iterable data : " << std::endl;
    for (auto i : item) {
      ProcessArgs(i);
    }
    arg_stream << "End of Iterable data." << std::endl;
  } else {
    arg_stream << "Data : " << item << std::endl;
  }
  arg_stream.close();
}

template <typename T>
inline void ProcessArgs(c10::optional<T>& item) {
  std::ofstream args_stream(GlobalConfig.arg_log_name, std::ios::app);
  args_stream << "Optional :";
  if (item.has_value()) {
    ProcessArgs(item.value());
  } else {
    args_stream << " Null " << std::endl;
  }
  args_stream.close();
}

template <typename T>
inline void ProcessArgs(torch::List<T>& item) {
  std::ofstream args_stream(GlobalConfig.arg_log_name, std::ios::app);
  args_stream << "Torch List : Size = " << item.size() << " : " << std::endl;
  for (T i : item) {
    ProcessArgs(i);
  }
  args_stream << "End of List" << std::endl;
  args_stream.close();
}

template <typename T>
inline void ProcessArgs(c10::OptionalArrayRef<T>& item) {
  std::ofstream args_stream(GlobalConfig.arg_log_name, std::ios::app);
  args_stream << " OptionalArray : ";
  if (item.has_value()) {
    ProcessArgs(item.value());
  } else {
    args_stream << " Null " << std::endl;
  }
  args_stream.close();
}

template <typename T>
inline void ProcessArgs(c10::IListRef<T>& item) {
  std::ofstream args_stream(GlobalConfig.arg_log_name, std::ios::app);
  args_stream << "IListRef : ";
  const auto& materialized = item.materialize();
  for (int idx = 0; idx < materialized.size(); ++idx) {
    ProcessArgs(materialized[idx].get());
  }
  args_stream.close();
}

template <typename T, size_t S>
inline void ProcessArgs(std::array<T, S>& item) {
  std::ofstream args_stream(GlobalConfig.arg_log_name, std::ios::app);
  args_stream << "Array size = " << S << " : {" << std::endl;
  for (size_t i = 0; i < S; ++i) {
    ProcessArgs(item.at(i));
  }
  args_stream << "}  // End of Array" << std::endl;
  args_stream.close();
}

// Tuple:
template <int Index, typename... Args>
inline void ProcessArgs(std::tuple<Args...>& item) {
  if constexpr (Index < std::tuple_size<std::tuple<Args...>>::value) {
    std::ofstream args_stream(GlobalConfig.arg_log_name, std::ios::app);
    args_stream << "Item " << Index << " Of Tuple:" << std::endl;
    ProcessArgs(get<Index>(item));
    ProcessArgs<Index + 1>(item);
    args_stream.close();
    return;
  } else {
    return;
  }
}

template <typename... Args>
inline void ProcessArgs(std::tuple<Args...>& item) {
  std::ofstream args_stream(GlobalConfig.arg_log_name, std::ios::app);
  args_stream << "Tuple : size = "
              << std::tuple_size<std::tuple<Args...>>::value << " :{"
              << std::endl;
  ProcessArgs<0, Args...>(item);
  args_stream << "} // End of tuple." << std::endl;
  args_stream.close();
}

// End of tuple

template <>
inline void ProcessArgs(at::Generator& item) {
  std::ofstream arg_stream(GlobalConfig.arg_log_name, std::ios::app);
  arg_stream << "Generator : Seed = " << item.current_seed();
  arg_stream << " device = " << item.device() << std::endl;
  arg_stream.close();
}

template <>
inline void ProcessArgs(at::Stream& s) {
  std::ofstream args_stream(GlobalConfig.arg_log_name, std::ios::app);
  args_stream << "at::stream Skiped" << std::endl;
  args_stream.close();
}

template <>
inline void ProcessArgs(Tensor& item) {
  GlobalConfig.TensorProcessor(item);
}

template <>
inline void ProcessArgs(Scalar& item) {
  std::ofstream arg_stream(GlobalConfig.arg_log_name, std::ios::app);
  arg_stream << "Scalar : " << item << std::endl;
  /*if (item.isBoolean()) {
      arg_stream << item.to<bool>() << std::endl;
  } else if (item.isIntegral(false)) {
      arg_stream << item.to<int>() << std::endl;
  } else {
      arg_stream << item.to<float>() << std::endl;
  }*/
  arg_stream.close();
}

template <>
inline void ProcessArgs(c10::QScheme& item) {
  std::ofstream arg_stream(GlobalConfig.arg_log_name, std::ios::app);
  arg_stream << "QScheme : " << c10::toString(item) << std::endl;
  arg_stream.close();
}

// traversal for multi args.
template <typename A>
inline void TraversalItems(A arg1) {
  ProcessArgs(arg1);
}
template <typename A, typename... Args>
inline void TraversalItems(A arg1, Args... args) {
  ProcessArgs(arg1);
  TraversalItems(args...);
}

// A hack used to suppress warnings when overriding operators for CPU backend
class SuppressOpOverrideWarningHandler : public c10::WarningHandler {
 public:
  void process(const c10::Warning& warning) {}
};

static c10::WarningHandler* GetSuppressOpOverrideHandler() {
  static SuppressOpOverrideWarningHandler handler;
  return &handler;
};

inline int EnterSuppressingOpOverrideWarning() {
  c10::WarningUtils::set_warning_handler(GetSuppressOpOverrideHandler());
  return 1;
}

inline int ExitSuppressingOpOverrideWarning() {
  c10::WarningUtils::set_warning_handler(nullptr);
  return 1;
}

#define CAT_HELPER(a, b) a##b
#define CATVARS(a, b) CAT_HELPER(a, b)
#define WARNING_VARNAME(var) CATVARS(var, __COUNTER__)

#define OVERRIDE_SELECTIVE_OPERATOR_REGISTER_WITHOUT_WARNING(op, fn) \
  static int WARNING_VARNAME(enter_selective_warning) =              \
      EnterSuppressingOpOverrideWarning();                           \
  TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {                       \
    m.impl(TORCH_SELECTIVE_NAME(op), TORCH_FN(fn));                  \
  }                                                                  \
  static int WARNING_VARNAME(exit_selective_warning) =               \
      ExitSuppressingOpOverrideWarning();

#define OVERRIDE_CPU_OPERATOR_REGISTER_WITHOUT_WARNING(op, fn) \
  static int WARNING_VARNAME(enter_cpu_warning) =              \
      EnterSuppressingOpOverrideWarning();                     \
  TORCH_LIBRARY_IMPL(aten, CPU, m) {                           \
    m.impl(op, TORCH_FN(fn));                                  \
  }                                                            \
  static int WARNING_VARNAME(exit_cpu_warning) =               \
      ExitSuppressingOpOverrideWarning();

} // namespace musa
} // namespace at

#endif // TORCH_MUSA_CSRC_UTILS_REGISTER_WRAPPER_H
