#include "torch_musa/csrc/utils/register_wrapper.h"
namespace at {
namespace musa {

Config GlobalConfig = Config();

Config::Config() {
  LoadEnv(); // Init from environment.
  InitConfig(); // Not supported.
}

void Config::InitConfig() {
  // Not supported.
}

void Config::set_enabled(bool flag) {
  enabled_ = flag;
}

void Config::set_dir(const std::string& dir) {
  base_dir_created_ = false;
  base_dir_ = dir;
}

/*
LoadEnv DO NOT short-circuit, since users can reset enable flag in python API.
Thus we have to load full OP list and OP level to avoid unexpected errors.
*/
void Config::LoadEnv() {
  char* env_enabled = std::getenv("TORCH_MUSA_OP_DEBUG");
  // Not unset, and not OFF.
  if (nullptr == env_enabled) {
    enabled_ = false;
  } else if (
      strcmp(env_enabled, "OFF") == 0 || strcmp(env_enabled, "0") == 0 ||
      strcmp(env_enabled, "off") == 0 || strcmp(env_enabled, "false") == 0) {
    enabled_ = false;
  } else {
    enabled_ = true;
  }

  char* env_level = std::getenv("TORCH_MUSA_OP_DEBUG_LEVEL");
  /*
  Level list =
              Lv1: 1 / Info / info / i / I => Only name, type, no details.
              Lv2: 2 / Sign / sign / signature / Signature / s / S => Signature
  of tensors. Lv3: 3 / Detail / detail / det / Det / d / D => First X data of
  tensors. X default is 50. Lv4: 4 / Full / full / f / F => Full tensor infos
  with full data. Lv5: 5 / SD / sd => L2 + L3, signature to log, and data to
  file. Lv6: 6 / SF / sf => L2 + L4, signature to log, and full data to file.
      Level can only set to 1 .. 6;
      TODO: Only support "1", "2", "3", "4", "5", "6" at present.
  */
  if (nullptr == env_level) {
    level_ = 1; //  default is Lv1;
  } else {
    std::stringstream ss1;
    // TODO(yueran.tang): Add int assert here.
    ss1 << env_level;
    ss1 >> level_;
  }

  // get env: tensor detail length
  char* env_length = std::getenv("TORCH_MUSA_OP_DEBUG_LENGTH");
  if (nullptr == env_length) {
    tensor_max_size_ = 50;
  } else {
    std::stringstream ss2;
    ss2 << env_length;
    ss2 >> tensor_max_size_;
  }

  // get env: OP white list.
  char* env_op_list = std::getenv("TORCH_MUSA_OP_DEBUG_LIST");
  if (nullptr == env_op_list) {
    has_op_white_list_ = false;
  } else {
    has_op_white_list_ = true;
    std::string last_op = "";
    for (int i = 0; i < strlen(env_op_list); ++i) {
      if (env_op_list[i] == ',' && last_op.length() > 0) {
        op_white_list_.push_back(last_op);
        last_op = "";
      } else {
        last_op += std::tolower(env_op_list[i]);
      }
    }
    if (last_op != "") {
      op_white_list_.push_back(last_op);
    }
  }

  // get env: OP black list.
  env_op_list = std::getenv("TORCH_MUSA_OP_DEBUG_BLACK_LIST");
  if (nullptr == env_op_list) {
    has_op_black_list_ = false;
  } else {
    has_op_black_list_ = true;
    std::string last_op = "";
    for (int i = 0; i < strlen(env_op_list); ++i) {
      if (env_op_list[i] == ',' && last_op.length() > 0) {
        op_black_list_.push_back(last_op);
        last_op = "";
      } else {
        last_op += std::tolower(env_op_list[i]);
      }
    }
    if (last_op != "") {
      op_black_list_.push_back(last_op);
    }
  }
  if (has_op_black_list_ && has_op_white_list_) {
    std::cerr << "It's not allowed to use TORCH_MUSA_OP_DEBUG_LIST and"
              << " TORCH_MUSA_OP_DEBUG_BLACK_LIST at same time." << std::endl;
  }
  TORCH_CHECK(
      !(has_op_black_list_ && has_op_white_list_),
      "It's not allowed to use TORCH_MUSA_OP_DEBUG_LIST and",
      " TORCH_MUSA_OP_DEBUG_BLACK_LIST at same time.");

  // get dir.
  char* env_op_dir = std::getenv("TORCH_MUSA_OP_DEBUG_DIR");
  if (nullptr == env_op_dir) {
    base_dir_ = "./DEBUG_DIR";
  } else {
    base_dir_ = env_op_dir;
  }
  return; // Done.
}

bool Config::IsOpEnabled(const char* yaml, const char* func) {
  if (!enabled_) {
    return false;
  }
  if ((!has_op_white_list_) &&
      (!has_op_black_list_)) { // no list means enable every func.
    return true;
  }
  std::stringstream ss;
  ss << yaml;
  std::string yaml_s;
  ss >> yaml_s;
  ss << func;
  std::string func_s;
  ss >> func_s;
  std::transform(
      yaml_s.begin(), yaml_s.end(), yaml_s.begin(), [](unsigned char c) {
        return std::tolower(c);
      });
  std::transform(
      func_s.begin(), func_s.end(), func_s.begin(), [](unsigned char c) {
        return std::tolower(c);
      });
  if (has_op_white_list_) {
    for (auto keys : op_white_list_) {
      if (yaml_s.find(keys) != std::string::npos ||
          func_s.find(keys) != std::string::npos) {
        return true;
      }
    }
    return false;
  } else if (has_op_black_list_) {
    for (auto keys : op_black_list_) {
      if (yaml_s.find(keys) != std::string::npos ||
          func_s.find(keys) != std::string::npos) {
        return false;
      }
    }
    return true;
  }
  TORCH_CHECK(
      false, "Unexpected status of op black or white list in debug tools.")
  return false;
}

/*
Calculate the signature of tensor.
The input Tensor must be a CPU Tensor with available data type.

For
*/
TensorSignature calc_tensor_signature(at::Tensor& t) {
  TensorSignature Sign;
  Sign.tensor_size = t.numel();
  Sign.has_nan = false;
  Sign.has_inf = false;
  Sign.has_num = false;
  if (t.scalar_type() == at::kBool) {
    Sign.true_num = t.sum().item().to<int>();
    Sign.is_bool_tensor = true;
    Sign.has_num = true;
    return Sign;
  }
  Sign.is_bool_tensor = false;
  Sign.true_num = 0;
  // Check if nan or inf.
  Sign.max_value = -FLT_MAX;
  Sign.min_value = FLT_MAX;
  Sign.nearest_zero_pos = FLT_MAX;
  Sign.nearest_zero_neg = -FLT_MAX;
  Sign.sum = 0;
  Sign.avg = 0;
  Sign.abs_sum = 0;
  Sign.standard_deviation = 0;
  for (int i = 0; i < t.numel(); ++i) {
    auto item = t[i].item().to<float>();
    // auto fp_item = static_cast<float>(item);
    if (std::isinf(item)) {
      Sign.has_inf = true;
      continue;
    }
    if (std::isnan(item)) {
      Sign.has_nan = true;
      continue;
    }
    Sign.has_num = true;
    Sign.max_value = item > Sign.max_value ? item : Sign.max_value;
    Sign.min_value = item < Sign.min_value ? item : Sign.min_value;
    Sign.nearest_zero_pos = (item > 0 && item < Sign.nearest_zero_pos)
        ? item
        : Sign.nearest_zero_pos;
    Sign.nearest_zero_neg = (item < 0 && item > Sign.nearest_zero_neg)
        ? item
        : Sign.nearest_zero_neg;
    Sign.sum += item;
    Sign.abs_sum += item > 0 ? item : -item;
  }
  // If no pos or neg value, nearest zero change to the opposite.
  if (Sign.nearest_zero_neg == -FLT_MAX) {
    Sign.nearest_zero_neg = Sign.nearest_zero_pos;
  }
  if (Sign.nearest_zero_pos == FLT_MAX) {
    Sign.nearest_zero_pos = Sign.nearest_zero_neg;
  }
  if (!Sign.has_num)
    return Sign; // Skip STD;
  Sign.avg = Sign.sum / t.numel();
  for (int i = 0; i < t.numel(); ++i) {
    auto item = t[i].item().to<float>();
    if (std::isinf(item) || std::isnan(item))
      continue;
    Sign.standard_deviation += (item - Sign.avg) * (item - Sign.avg);
  }
  Sign.standard_deviation = std::sqrt(Sign.standard_deviation / t.numel());
  return Sign;
}

// std::ostream& TensorSignature::operator << (std::ostream& os,
// TensorSignature& ts) {
std::ostream& operator<<(std::ostream& os, const TensorSignature& ts) {
  os << "Tensor Signature: " << std::endl;
  if (ts.is_bool_tensor) {
    os << "BoolTensor with : " << ts.true_num << " of True." << std::endl;
  } else {
    if (ts.has_nan) {
      os << "Warning: NaN Detected." << std::endl;
    }
    if (ts.has_inf) {
      os << "Warning: Inf Detected." << std::endl;
    }
    if (!ts.has_num && ts.tensor_size > 0) {
      os << "Warning: Tensor only has NaN or Inf." << std::endl;
      return os;
    }
    if (ts.tensor_size == 0) {
      return os;
    }
    os << "Max, Min & Avg Value : " << ts.max_value << " , " << ts.min_value
       << " , " << ts.avg << std::endl;
    os << "Value Nearest Zero : " << ts.nearest_zero_pos << " : "
       << ts.nearest_zero_neg << std::endl;
    os << "Sum, AbsSum, STD : " << ts.sum << " , " << ts.abs_sum << " , "
       << ts.standard_deviation << std::endl;
  }
  os << "======================================" << std::endl;
  return os;
}

void Config::TensorProcessor(const at::Tensor& t) {
  CreateTensorStream();
  std::ofstream arg_stream(arg_log_name, std::ios::app);
  std::ofstream log_stream(base_log, std::ios::app);
  if (t.defined() == false || t.data_ptr() == nullptr) {
    arg_stream << " Tensor : Undefined." << std::endl;
    log_stream << std::setfill(' ') << std::setw(split_dirs_.size() * 2 - 1)
               << ""
               << " Tensor : Undefined." << std::endl;
    arg_stream.close();
    log_stream.close();
    CloseTensorStream();
    return;
  }
  // Base info
  auto size = t.numel();
  auto shape = t.sizes();
  auto stride = t.strides();
  auto dtype = t.scalar_type();
  auto dev = t.device();
  arg_stream << "Tensor size : " << size << " shape : " << shape
             << " strides : " << stride << " dtype : " << dtype
             << " device : " << dev << std::endl;

  log_stream << std::setfill(' ') << std::setw(split_dirs_.size() * 2 - 1) << ""
             << "Tensor size : " << size << " shape : " << shape
             << " strides : " << stride << " dtype : " << dtype
             << " device : " << dev << std::endl;
  if (level_ == 1) {
    arg_stream.close();
    log_stream.close();
    CloseTensorStream();
    return; // break first to avoid cpu copy.
  }
  if (t.is_complex()) {
    log_stream << std::setfill(' ') << std::setw(split_dirs_.size() * 2 - 1)
               << ""
               << "[Warning] Unsupported Dtype: Complex." << std::endl;
    arg_stream << "[Warning] Unsupported Dtype: Complex." << std::endl;
    arg_stream.close();
    log_stream.close();
    CloseTensorStream();
    return;
  }
  // TO CPU. Warning: to cpu is really slow and don't use it usually.
  at::Tensor cpu_tensor;
  if (dtype == at::kFloat || dtype == at::kInt || dtype == at::kDouble ||
      dtype == at::kLong) {
    cpu_tensor = t.cpu().detach().clone().reshape({size});
  } else {
    // for not base type tensor, trans to float.
    cpu_tensor = t.to(at::kFloat).cpu().detach().clone().reshape({size});
  }
  // Signature:
  if (level_ == 2 || level_ >= 5) { // lv 2,5,6
    auto signature = calc_tensor_signature(cpu_tensor);
    arg_stream << signature << std::endl;
    if (signature.has_nan) {
      log_stream << std::setfill(' ') << std::setw(split_dirs_.size() * 2 - 1)
                 << ""
                 << "[Warning] NaN detected in Tensor." << std::endl;
    }
    if (signature.has_inf) {
      log_stream << std::setfill(' ') << std::setw(split_dirs_.size() * 2 - 1)
                 << ""
                 << "[Warning] Inf detected in Tensor." << std::endl;
    }
    if (!signature.has_num && signature.tensor_size > 0) {
      log_stream << std::setfill(' ') << std::setw(split_dirs_.size() * 2 - 1)
                 << ""
                 << "[Error] Values in tensor is all of inf or nan."
                 << std::endl;
    }
  }
  if (level_ == 3 || level_ == 5 || level_ == 6) { // out first N or all items
    std::ofstream tensor_stream(tensor_file_name, std::ios::app);
    size_t max_num = level_ == 6 ? size : tensor_max_size_;
    for (int i = 0; i < max_num && i < size; ++i) {
      tensor_stream << cpu_tensor[i].item() << std::endl;
    }
    tensor_stream.close();
  }
  arg_stream.close();
  log_stream.close();
}

void Config::TryCreateBaseDir() {
  if (base_dir_created_) {
    return;
  }
  if (base_dir_.back() == '/') {
    base_dir_.pop_back();
  }
  if (access(base_dir_.c_str(), 0) == -1) {
    mkdir(base_dir_.c_str(), 0755);
    base_dir_created_ = true;
    if (base_dir_ == ".") {
      base_dir_ = ""; // set to null and suffix at end.
    }
  } else {
    auto temp_dir = base_dir_;
    int dir_suffix = 1; // append suffix to base_dir_;
    do {
      temp_dir = base_dir_ + "_" + std::to_string(dir_suffix);
      dir_suffix++;
    } while (access(temp_dir.c_str(), 0) == 0);
    base_dir_ = temp_dir; // base_dir_ = "{base_dir_}_{N}"
    mkdir(base_dir_.c_str(), 0755);
    base_dir_created_ = true;
  }
  base_log = base_dir_ + "/full_log.txt";
  // gen dir vector:
  split_dirs_.push_back(base_dir_);
  dir_numbers_.push_back(0);
}

std::string gen_dir(std::vector<std::string> splits) {
  std::string dir = "";
  for (auto s : splits) {
    dir += s;
    dir += '/';
  }
  return dir;
}

/*
DirPath generator automation:
1. Create BaseDir.                                          <--
(TryCreateBaseDir) (once)
2. BaseNumber = [0]. Split = [base_dir_].
3. Get OP name.                                             <--
(create-kernel-stream)
4. Number [... , last ++]. Split = [base_dir_, ... , this_op]
5. Split = [base_dir_, ... , this_op_Number]
6. Creat save_dir_ = base_dir_ / ... / this_op_Number
7. Number = [..., last(+), 0] ; // Add 0 means ready for recursive op.
8. Log some info here.                                      <--
(SplitKernelIO)
9. Do calculation -> may call new OP -> goto 3.
10 Re Calculate save_dir_ (out of recursive) = base_dir_ / ... / this_op_Number
11 Get return value, write to save_dir_.
12 Destroy Number last 0 : Number = [..., last(+)]
13 Destroy Split last name : Split = [base_dir_, ... ]
14 End of this op. Finsh or goto 3 (next op)                <--
(CloseKernelStream)
*/

/*
Create kernel stream generate a dir named by OP-name and index.
And generate a log.txt file to record infos and non-tensor data.
*/
void Config::CreateKernelStream(const char* yaml, const char* name) {
  if (!base_dir_created_) {
    TryCreateBaseDir();
  }
  dir_numbers_.back() += 1; // the last number is ++ only!
  is_output_ = false;
  last_dir_ = "";
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(4) << dir_numbers_.back();
  ss >> last_dir_;
  last_dir_ += "_";
  last_dir_ += name; // last_dir_ is 000N_name.  with width = 4;
  dir_numbers_.push_back(0); // add 0 for inside operators.
  split_dirs_.push_back(last_dir_); // add last_dir_ == "N_name"
  // TODO(tyr): maybe std::filesystem can help this function.
  save_dir_ = gen_dir(split_dirs_);

  // The op dir is generated.
  mkdir(save_dir_.c_str(), 0755);
  // The save_file is generated.
  arg_log_name = save_dir_ + "log.txt";
  std::ofstream arg_stream(arg_log_name, std::ios::app);
  arg_stream << "Operator name : " << name << " Yaml: " << yaml << std::endl;
  arg_stream.close();
  std::ofstream log_stream(base_log, std::ios::app);
  log_stream << std::setfill(' ') << std::setw(split_dirs_.size() * 2 - 2) << ""
             << "Operator : " << name << std::endl;
  log_stream.close();
}

/*
This function split input args and output, which will record an info in log.
And output tensor name will add suffix "_output"
*/
void Config::SplitKernelIO() {
  is_output_ = true;
  std::ofstream log_stream(arg_log_name, std::ios::app);
  log_stream << " ------------output----------------" << std::endl;
  log_stream.close();
}

/*
 */
void Config::CloseKernelStream() {
  split_dirs_.pop_back(); // remove this op.
  dir_numbers_.pop_back(); // remove the end 0.
  // re-calculate save_dir_ and log file for parent op.
  save_dir_ = gen_dir(split_dirs_);
  arg_log_name = save_dir_ + "log.txt";
}

/*
This function create a new IO stream to return the output of tensor.
IO stream will closed by close_tensor_out;
*/
void Config::CreateTensorStream() {
  dir_numbers_.back() += 1;
  if (is_output_) {
    tensor_file_name = save_dir_ + "output_tensor_";
  } else {
    tensor_file_name = save_dir_ + "tensor_";
  }
  std::stringstream ss;
  ss << dir_numbers_.back();
  std::string number_str;
  ss >> number_str;
  tensor_file_name += number_str;
}

/*
This function ends a process of tensor.
*/
void Config::CloseTensorStream() {
  // do nothing.
}

} // namespace musa
} // namespace at