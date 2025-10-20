#include <ATen/Config.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fft_c2c_cpu_dispatch.h>
#include <ATen/ops/_fft_r2c_cpu_dispatch.h>
#include <ATen/ops/conj.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/view_as_complex.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/zeros.h>
#endif

#include <mufft.h>
#include <mufftXt.h>

#include "torch_musa/csrc/aten/musa/MuFFTPlanCache.h"
#include "torch_musa/csrc/aten/musa/MuFFTUtils.h"

namespace at {
namespace musa {
using namespace at::native::detail;
using at::native::fft_norm_mode;
using at::native::MUFFT_CHECK;

namespace {

template <typename Stream, typename T>
static Stream& WriteOpt(Stream& s, const optional<T>& value) {
  if (value) {
    s << *value;
  } else {
    s << "None";
  }
  return s;
}

Tensor CpuAwareFFTReal2Complex(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided) {
  const OptionalDeviceGuard device_guard(self.device());

  auto cpu_self =
      at::empty(self.sizes(), self.options().device(DeviceType::CPU));
  cpu_self.copy_(self);

  Tensor cpu_out = at::cpu::_fft_r2c(cpu_self, dim, normalization, onesided);

  return cpu_out;
}

Tensor CpuAwareFFTComplex2Complex(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided) {
  const OptionalDeviceGuard device_guard(self.device());

  auto cpu_self =
      at::empty(self.sizes(), self.options().device(DeviceType::CPU));
  cpu_self.copy_(self);

  Tensor cpu_out = at::cpu::_fft_c2c(cpu_self, dim, normalization, onesided);

  return cpu_out;
}

} // anonymous namespace

Tensor StftCenter(
    const Tensor& self,
    const int64_t n_fft,
    const optional<int64_t> hop_lengthOpt,
    const optional<int64_t> win_lengthOpt,
    const c10::optional<Tensor>& window_opt,
    const bool center,
    c10::string_view mode,
    const bool normalized,
    const optional<bool> onesidedOpt,
    const optional<bool> return_complexOpt) {
  c10::MaybeOwned<Tensor> window_maybe_owned =
      at::borrow_from_optional_tensor(window_opt);
  const Tensor& window = *window_maybe_owned;

#define REPR(SS)                                                          \
  SS << "stft(" << self.toString() << self.sizes() << ", n_fft=" << n_fft \
     << ", hop_length=" << hop_length << ", win_length=" << win_length    \
     << ", window=";                                                      \
  if (window.defined()) {                                                 \
    SS << window.toString() << "{" << window.sizes() << "}";              \
  } else {                                                                \
    SS << "None";                                                         \
  }                                                                       \
  SS << ", normalized=" << normalized << ", onesided=";                   \
  WriteOpt(SS, onesidedOpt) << ", return_complex=";                       \
  WriteOpt(SS, return_complexOpt) << ") "

  TORCH_CHECK(
      !window.defined() || window.device() == self.device(),
      "stft input and window must be on the same device but got self on ",
      self.device(),
      " and window on ",
      window.device())

  // default_init hop_length and win_length
  auto hop_length = hop_lengthOpt.value_or(n_fft >> 2);
  auto win_length = win_lengthOpt.value_or(n_fft);
  const bool return_complex = return_complexOpt.value_or(
      self.is_complex() || (window.defined() && window.is_complex()));
  if (!return_complex) {
    TORCH_CHECK(
        return_complexOpt.has_value(),
        "stft requires the return_complex parameter be given for real inputs, "
        "and will further require that return_complex=True in a future "
        "PyTorch release.");

    TORCH_WARN_ONCE(
        "stft with return_complex=False is deprecated. In a future pytorch "
        "release, stft will return complex tensors for all inputs, and "
        "return_complex=False will raise an error.\n"
        "Note: you can still call torch.view_as_real on the complex output to "
        "recover the old return format.");
  }

  if (!at::isFloatingType(self.scalar_type()) &&
      !at::isComplexType(self.scalar_type())) {
    std::ostringstream ss;
    REPR(ss) << ": expected a tensor of floating point or complex values";
    AT_ERROR(ss.str());
  }
  if (self.dim() > 2 || self.dim() < 1) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D or 2D tensor";
    AT_ERROR(ss.str());
  }
  Tensor input = self;
  if (self.dim() == 1) {
    input = input.unsqueeze(0);
  }
  if (center) {
    const auto input_shape = input.sizes();
    const auto input_dim = input_shape.size();
    const auto extra_dims = std::max(size_t{3}, input_dim) - input_dim;
    const auto pad_amount = n_fft / 2;

    DimVector extended_shape(extra_dims, 1);
    extended_shape.append(input_shape.begin(), input_shape.end());
    input = at::pad(input.view(extended_shape), {pad_amount, pad_amount}, mode);
    input = input.view(IntArrayRef(input.sizes()).slice(extra_dims));
  }
  int64_t batch = input.size(0);
  int64_t len = input.size(1);
  if (n_fft <= 0 || n_fft > len) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < n_fft < " << len
             << ", but got n_fft=" << win_length;
    AT_ERROR(ss.str());
  }
  if (hop_length <= 0) {
    std::ostringstream ss;
    REPR(ss) << ": expected hop_length > 0, but got hop_length=" << hop_length;
    AT_ERROR(ss.str());
  }
  if (win_length <= 0 || win_length > n_fft) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < win_length <= n_fft, but got win_length="
             << win_length;
    AT_ERROR(ss.str());
  }
  if (window.defined() && (window.dim() != 1 || window.size(0) != win_length)) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D window tensor of size equal to win_length="
             << win_length << ", but got window with size " << window.sizes();
    AT_ERROR(ss.str());
  }
#undef REPR
  auto window_ = window;
  if (win_length < n_fft) {
    // pad center
    auto left = (n_fft - win_length) / 2;
    if (window.defined()) {
      window_ = at::zeros({n_fft}, window.options());
      window_.narrow(0, left, win_length).copy_(window);
    } else {
      window_ = at::zeros({n_fft}, self.options());
      window_.narrow(0, left, win_length).fill_(1);
    }
  }
  int64_t n_frames = 1 + (len - n_fft) / hop_length;

  // time2col
  input = input.as_strided(
      {batch, n_frames, n_fft},
      {input.stride(0), hop_length * input.stride(1), input.stride(1)});
  if (window_.defined()) {
    input = input.mul(window_);
  }

  // FFT and transpose to get (batch x fft_size x num_frames)
  const bool complex_fft = input.is_complex();
  const auto onesided = onesidedOpt.value_or(!complex_fft);

  const at::native::fft_norm_mode norm = normalized
      ? at::native::fft_norm_mode::by_root_n
      : at::native::fft_norm_mode::none;
  Tensor out_cpu;

  if (complex_fft) {
    TORCH_CHECK(
        !onesided, "Cannot have onesided output if window or input is complex");
    out_cpu = CpuAwareFFTComplex2Complex(
        input,
        input.dim() - 1,
        static_cast<int64_t>(norm),
        /*forward=*/true);
  } else {
    out_cpu = CpuAwareFFTReal2Complex(
        input, input.dim() - 1, static_cast<int64_t>(norm), onesided);
  }

  out_cpu.transpose_(1, 2);
  Tensor out = at::view_as_real(out_cpu).to(self.device()).contiguous();

  if (self.dim() == 1) {
    out.squeeze_(0);
  }

  if (return_complex) {
    return at::view_as_complex(out);
  } else {
    return out;
  }
}

Tensor Stft(
    const Tensor& self,
    const int64_t n_fft,
    const optional<int64_t> hop_lengthOpt,
    const optional<int64_t> win_lengthOpt,
    const c10::optional<Tensor>& window_opt,
    const bool normalized,
    const optional<bool> onesidedOpt,
    const optional<bool> return_complexOpt) {
  return StftCenter(
      self,
      n_fft,
      hop_lengthOpt,
      win_lengthOpt,
      window_opt,
      /*center=*/false,
      /*mode=*/"constant",
      normalized,
      onesidedOpt,
      return_complexOpt);
}

namespace {

static std::vector<std::unique_ptr<MuFFTParamsLRUCache>> plan_caches;
static std::mutex plan_caches_mutex;
constexpr int64_t mufft_max_ndim = 3;

static void exec_mufft_plan(
    const MuFFTConfig& config,
    void* in_data,
    void* out_data,
    bool forward) {
  auto& plan = config.plan();
  MUFFT_CHECK(mufftXtExec(
      plan, in_data, out_data, forward ? MUFFT_FORWARD : MUFFT_INVERSE));
}

static inline MuFFTParamsLRUCache& mufft_get_plan_cache(
    DeviceIndex device_index) {
  std::lock_guard<std::mutex> guard(plan_caches_mutex);

  AT_ASSERT(device_index >= 0);

  if (device_index >= static_cast<int64_t>(plan_caches.size())) {
    plan_caches.resize(device_index + 1);
  }

  if (!plan_caches[device_index]) {
    plan_caches[device_index] = std::make_unique<MuFFTParamsLRUCache>();
  }

  return *plan_caches[device_index];
  // Execute a general fft operation (can be c2c, onesided r2c or onesided c2r)
}

static const Tensor& _exec_fft(
    Tensor& out,
    const Tensor& self,
    IntArrayRef out_sizes,
    IntArrayRef dim,
    bool forward) {
  const auto ndim = self.dim();
  const int64_t signal_ndim = dim.size();
  const auto batch_dims = ndim - signal_ndim;

  // Permute dimensions so batch dimensions come first, and in stride order
  // This maximizes data locality when collapsing to a single batch dimension
  DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), int64_t{0});

  c10::SmallVector<bool, kDimVectorStaticSize> is_transformed_dim(ndim);
  for (const auto& d : dim) {
    is_transformed_dim[d] = true;
  }
  auto batch_end =
      std::partition(dim_permute.begin(), dim_permute.end(), [&](int64_t d) {
        return !is_transformed_dim[d];
      });
  auto self_strides = self.strides();
  std::sort(dim_permute.begin(), batch_end, [&](int64_t a, int64_t b) {
    return self_strides[a] > self_strides[b];
  });
  std::copy(dim.cbegin(), dim.cend(), batch_end);
  auto input = self.permute(dim_permute);

  // Collapse batch dimensions into a single dimension
  DimVector batched_sizes(signal_ndim + 1);
  batched_sizes[0] = -1;
  std::copy(
      input.sizes().cbegin() + batch_dims,
      input.sizes().cend(),
      batched_sizes.begin() + 1);
  input = input.reshape(batched_sizes);

  const auto batch_size = input.sizes()[0];
  DimVector signal_size(signal_ndim + 1);
  signal_size[0] = batch_size;
  for (const auto i : c10::irange(signal_ndim)) {
    auto in_size = input.sizes()[i + 1];
    auto out_size = out_sizes[dim[i]];
    signal_size[i + 1] = std::max(in_size, out_size);
    TORCH_INTERNAL_ASSERT(
        in_size == signal_size[i + 1] ||
        in_size == (signal_size[i + 1] / 2) + 1);
    TORCH_INTERNAL_ASSERT(
        out_size == signal_size[i + 1] ||
        out_size == (signal_size[i + 1] / 2) + 1);
  }

  batched_sizes[0] = batch_size;
  DimVector batched_out_sizes(batched_sizes.begin(), batched_sizes.end());
  for (const auto i : c10::irange(dim.size())) {
    batched_out_sizes[i + 1] = out_sizes[dim[i]];
  }
  out.resize_(batched_out_sizes, MemoryFormat::Contiguous);

  // Create the transform plan (either from cache or locally)
  const auto value_type = c10::toRealValueType(input.scalar_type());
  auto fft_type = GetMuFFTTransformType(input.is_complex(), out.is_complex());
  MuFFTParams Params(
      input.strides(), out.strides(), signal_size, fft_type, value_type);
  MuFFTParamsLRUCache& plan_cache =
      mufft_get_plan_cache(input.device().index());
  std::unique_lock<std::mutex> guard(plan_cache.mutex, std::defer_lock);
  std::optional<MuFFTConfig> uncached_plan;
  const MuFFTConfig* config = nullptr;

  bool use_caching = true;

  if (use_caching && plan_cache.max_size() > 0) {
    guard.lock();
    if (plan_cache.max_size() > 0) { // check again after acquiring the lock
      config = &plan_cache.lookup(Params);
    }
  }

  if (config == nullptr) {
    uncached_plan.emplace(Params);
    config = &uncached_plan.value();
  }

  auto& plan = config->plan();

  if (config->should_clone_input()) {
    input = input.clone(MemoryFormat::Contiguous);
  }

  // prepare mufft for execution
  MUFFT_CHECK(mufftSetStream(plan, at::musa::getCurrentMUSAStream()));
  auto workspace = at::empty(
      {config->workspace_size()}, at::device(at::kMUSA).dtype(at::kByte));
  MUFFT_CHECK(mufftSetWorkArea(plan, workspace.mutable_data_ptr()));

  exec_mufft_plan(
      *config,
      const_cast<void*>(input.const_data_ptr()),
      out.data_ptr(),
      forward);

  // Inplace reshaping to original batch shape and inverting the dimension
  // permutation
  DimVector out_strides(ndim);
  int64_t batch_numel = 1;
  for (int64_t i = batch_dims - 1; i >= 0; --i) {
    out_strides[dim_permute[i]] = batch_numel * out.strides()[0];
    batch_numel *= out_sizes[dim_permute[i]];
  }
  for (const auto i : c10::irange(batch_dims, ndim)) {
    out_strides[dim_permute[i]] = out.strides()[1 + (i - batch_dims)];
  }
  return out.as_strided_(out_sizes, out_strides, out.storage_offset());
}

bool use_optimized_mufft_path(IntArrayRef dim) {
  // For performance reason, when dim starts with (0, 1), do not use the
  // optimized path.
  if (dim.size() > mufft_max_ndim ||
      (dim.size() >= 2 && dim[0] == 0 && dim[1] == 1)) {
    return false;
  } else {
    return true;
  }
}

double _fft_normalization_scale(
    int64_t normalization,
    IntArrayRef sizes,
    IntArrayRef dims) {
  auto norm = static_cast<fft_norm_mode>(normalization);
  if (norm == fft_norm_mode::none) {
    return 1.0;
  }

  int64_t signal_numel = 1;
  for (auto dim : dims) {
    signal_numel *= sizes[dim];
  }
  const double scale_denom = (norm == fft_norm_mode::by_root_n)
      ? std::sqrt(signal_numel)
      : static_cast<double>(signal_numel);
  return 1.0 / scale_denom;
}

/*
  FIXME: The MUSA backend does not support complex tensors for these operations
         (mul, copy_, ...), so we currently fallback to CPU tensor iteration.
  TODO: Implement these operations directly on MUSA tensors once supported.
*/
Tensor _fft_apply_normalization(
    const Tensor& self,
    int64_t normalization,
    IntArrayRef sizes,
    IntArrayRef dims) {
  auto scale = _fft_normalization_scale(normalization, sizes, dims);
  Tensor tmp;
  if (self.is_complex() && self.device().is_musa()) {
    tmp = self.cpu().mul(static_cast<double>(scale));
    tmp = tmp.to(self.device());
  } else {
    return (scale == 1.0) ? self : self.mul_(scale);
  }
  return tmp;
}

ScalarType promote_type_fft(
    ScalarType type,
    bool require_complex,
    Device device) {
  if (at::isComplexType(type)) {
    return type;
  }
  // Promote integral to default float type
  if (!at::isFloatingType(type)) {
    type = c10::typeMetaToScalarType(c10::get_default_dtype());
  }

  const bool maybe_support_half = (device.is_musa() || device.is_meta());
  if (maybe_support_half) {
    TORCH_CHECK(
        type == kHalf || type == kFloat || type == kDouble,
        "Unsupported dtype ",
        type);
  } else {
    TORCH_CHECK(type == kFloat || type == kDouble, "Unsupported dtype ", type);
  }

  if (!require_complex) {
    return type;
  }

  // Promote to complex
  switch (type) {
    case kHalf:
      return kComplexHalf;
    case kFloat:
      return kComplexFloat;
    case kDouble:
      return kComplexDouble;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unhandled dtype");
  }
}

// Promote a tensor's dtype according to promote_type_fft
Tensor promote_tensor_fft(const Tensor& t, bool require_complex = false) {
  auto cur_type = t.scalar_type();
  auto new_type = promote_type_fft(cur_type, require_complex, t.device());
  return (cur_type == new_type) ? t : t.to(new_type);
}

Tensor resize_fft_input(Tensor x, IntArrayRef dims, SymIntArrayRef sizes) {
  TORCH_INTERNAL_ASSERT(dims.size() == sizes.size());
  bool must_copy = false;
  auto x_sizes = x.sym_sizes();
  SymDimVector pad_amount(x_sizes.size() * 2);
  for (const auto i : c10::irange(dims.size())) {
    if (sizes[i] == -1) {
      continue;
    }

    if (x_sizes[dims[i]] < sizes[i]) {
      must_copy = true;
      auto pad_idx = pad_amount.size() - 2 * dims[i] - 1;
      pad_amount[pad_idx] = sizes[i] - x_sizes[dims[i]];
    }

    if (x_sizes[dims[i]] > sizes[i]) {
      x = x.slice_symint(dims[i], 0, sizes[i]);
    }
  }

  // Only call pad if necessary since pad copies the entire tensor
  return must_copy ? at::constant_pad_nd_symint(x, pad_amount) : x;
}

fft_norm_mode norm_from_string(
    std::optional<c10::string_view> norm,
    bool forward) {
  if (!norm || *norm == "backward") {
    return forward ? fft_norm_mode::none : fft_norm_mode::by_n;
  }

  if (*norm == "forward") {
    return forward ? fft_norm_mode::by_n : fft_norm_mode::none;
  }

  if (*norm == "ortho") {
    return fft_norm_mode::by_root_n;
  }

  TORCH_CHECK(false, "Invalid normalization mode: \"", *norm, "\"")
}

Tensor& _fft_apply_normalization_out(
    Tensor& out,
    const Tensor& self,
    int64_t normalization,
    IntArrayRef sizes,
    IntArrayRef dims) {
  auto scale = _fft_normalization_scale(normalization, sizes, dims);
  return at::mul_out(out, self, c10::scalar_to_tensor(scale));
}

} // namespace

// n-dimensional real to complex FFT
Tensor _fft_r2c_musa(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided) {
  TORCH_CHECK(
      self.is_floating_point() &&
          (self.scalar_type() == at::ScalarType::Float ||
           self.scalar_type() == at::ScalarType::Half ||
           self.scalar_type() == at::ScalarType::BFloat16),
      "Expected dtype of input tensor is Float32, Half and BFloat16, but now it is ",
      self.scalar_type());

  const auto input_sizes = self.sizes();
  DimVector onesided_sizes(input_sizes.begin(), input_sizes.end());
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  onesided_sizes[last_dim] = last_dim_halfsize;
  IntArrayRef out_sizes = onesided ? onesided_sizes : input_sizes;

  const auto out_options =
      self.options().dtype(c10::toComplexType(self.scalar_type()));
  auto output = at::empty(out_sizes, out_options);

  const auto complex_size = 2 * self.element_size();
  const bool complex_aligned =
      (reinterpret_cast<std::uintptr_t>(self.const_data_ptr()) % complex_size ==
       0);
  auto working_tensor = self;
  if (!complex_aligned) {
    working_tensor = self.movedim(last_dim, -1)
                         .clone(MemoryFormat::Contiguous)
                         .movedim(-1, last_dim);
  }
  if (use_optimized_mufft_path(dim)) {
    _exec_fft(output, working_tensor, out_sizes, dim, /*forward=*/true);
  } else {
    {
      auto target_sizes = dim.size() == 1 ? out_sizes : onesided_sizes;
      _exec_fft(
          output, working_tensor, target_sizes, last_dim, /*forward=*/true);
      if (dim.size() > 1) {
        working_tensor = at::empty(out_sizes, out_options);
      }
    }

    // Then any remaining C2C transforms
    DimVector sorted_dims(dim.begin(), dim.end() - 1);
    while (!sorted_dims.empty()) {
      std::swap(output, working_tensor);

      // Resort dimensions every time as _exec_fft re-strides the output
      auto strides = working_tensor.strides();
      std::sort(
          sorted_dims.begin(), sorted_dims.end(), [&](int64_t a, int64_t b) {
            return strides[a] > strides[b];
          });

      const auto max_dims =
          std::min(static_cast<size_t>(mufft_max_ndim), sorted_dims.size());
      auto last_dims = IntArrayRef(sorted_dims)
                           .slice(sorted_dims.size() - max_dims, max_dims);

      // Intermediate results are always onesided
      _exec_fft(
          output, working_tensor, onesided_sizes, last_dims, /*forward=*/true);
      sorted_dims.resize(sorted_dims.size() - max_dims);
    }
  }

  // Only need to normalize the onesided slice since data in the other half is
  // overwritten
  // TODO: Support MUSA Complex Tensor
  auto output_cpu = output.cpu();
  auto out_slice = output_cpu.slice(last_dim, 0, last_dim_halfsize);
  _fft_apply_normalization(out_slice, normalization, input_sizes, dim);

  if (!onesided) {
    if (output.sizes()[last_dim] != out_sizes[last_dim]) {
      working_tensor.resize_(out_sizes, MemoryFormat::Contiguous);
      working_tensor.slice(last_dim, 0, last_dim_halfsize).copy_(output);
      output = std::move(working_tensor);
    }
    at::native::_fft_fill_with_conjugate_symmetry_(output, dim);
  }
  return output_cpu.to(self.device());
}

Tensor& _fft_r2c_musa_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided,
    Tensor& out) {
  auto result = _fft_r2c_musa(
      self, dim, static_cast<int64_t>(fft_norm_mode::none), /*onesided=*/true);
  if (onesided) {
    return _fft_apply_normalization_out(
        out, result, normalization, self.sizes(), dim);
  }

  at::native::resize_output(out, self.sizes());

  auto last_dim = dim.back();
  auto last_dim_halfsize = result.sizes()[last_dim];
  auto out_slice = out.slice(last_dim, 0, last_dim_halfsize);
  _fft_apply_normalization_out(
      out_slice, result, normalization, self.sizes(), dim);
  at::native::_fft_fill_with_conjugate_symmetry_(out, dim);
  return out;
}

// n-dimensional complex to complex FFT/IFFT
Tensor _fft_c2c_musa(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  TORCH_CHECK(self.is_complex());
  if (dim.empty()) {
    return self.clone();
  }

  auto out_sizes = self.sizes();
  auto output = at::empty(out_sizes, self.options());

  // Perform any number of C2C transforms
  DimVector sorted_dims(dim.begin(), dim.end());
  auto working_tensor = self;
  while (true) {
    // Sort dimensions every time as _exec_fft re-strides the output
    auto strides = working_tensor.strides();
    std::sort(
        sorted_dims.begin(), sorted_dims.end(), [&](int64_t a, int64_t b) {
          return strides[a] > strides[b];
        });

    const auto max_dims =
        std::min(static_cast<size_t>(mufft_max_ndim), sorted_dims.size());
    auto first_dims =
        IntArrayRef(sorted_dims).slice(sorted_dims.size() - max_dims, max_dims);

    _exec_fft(output, working_tensor, out_sizes, first_dims, forward);
    sorted_dims.resize(sorted_dims.size() - max_dims);

    if (sorted_dims.empty()) {
      break;
    }

    if (working_tensor.is_same(self)) {
      working_tensor = std::move(output);
      output = at::empty(out_sizes, self.options());
    } else {
      std::swap(output, working_tensor);
    }
  }

  return _fft_apply_normalization(output, normalization, out_sizes, dim);
}

Tensor& _fft_c2c_musa_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  auto result = _fft_c2c_musa(
      self, dim, static_cast<int64_t>(fft_norm_mode::none), forward);
  return _fft_apply_normalization_out(
      out, result, normalization, result.sizes(), dim);
}

// n-dimensional complex to real IFFT
Tensor _fft_c2r_musa(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t lastdim) {
  TORCH_CHECK(self.is_complex());
  auto in_sizes = self.sizes();
  DimVector out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes[dim.back()] = lastdim;

  auto output = at::empty(
      out_sizes,
      self.options().dtype(c10::toRealValueType(self.scalar_type())));

  if (use_optimized_mufft_path(dim)) {
    Tensor temp;
    // Complex to real FFTs may overwrite the input buffer, so must always clone
    // (gh-34551)
    /*
    TODO: temp = self.clone(MemoryFormat::Contiguous);
          MUSA does not support writing complex tensors (out) for this
    operation.
    */
    temp = self.cpu().clone().to(self.device());
    _exec_fft(output, temp, out_sizes, dim, /*forward=*/false);
  } else {
    // First complete any C2C transforms
    Tensor temp;
    if (dim.size() > 1) {
      temp = _fft_c2c_musa(
          self,
          dim.slice(0, dim.size() - 1),
          static_cast<int64_t>(fft_norm_mode::none),
          /*forward=*/false);
    } else {
      // Complex to real FFTs may overwrite the input buffer, so must always
      // clone (gh-34551)
      temp = self.clone(MemoryFormat::Contiguous);
    }

    // Finally, do a 1D C2R transform
    // TODO: could transform up to 2 other dims in the same muFFT operation
    _exec_fft(output, temp, out_sizes, dim.back(), /*forward=*/false);
  }

  return _fft_apply_normalization(output, normalization, out_sizes, dim);
}

Tensor& _fft_c2r_musa_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t lastdim,
    Tensor& out) {
  auto result = _fft_c2r_musa(
      self, dim, static_cast<int64_t>(fft_norm_mode::none), lastdim);
  return _fft_apply_normalization_out(
      out, result, normalization, result.sizes(), dim);
}

Tensor fft_c2c_maybe_out(
    c10::string_view fname,
    const Tensor& out,
    const Tensor& input,
    IntArrayRef dim,
    int64_t norm,
    bool forward) {
  if (out.defined()) {
    TORCH_CHECK(
        out.is_complex(),
        fname,
        " expects a complex output tensor, but got ",
        out.scalar_type());
    auto out_mut = out;
    return _fft_c2c_musa_out(input, dim, norm, forward, out_mut);
  }
  return _fft_c2c_musa(input, dim, norm, forward);
}

Tensor fft_c2r_maybe_out(
    c10::string_view fname,
    const Tensor& out,
    const Tensor& input,
    IntArrayRef dim,
    int64_t norm,
    SymInt last_dim_size) {
  // Support out argument if defined, otherwise call functional
  // variant so autograd works properly.
  if (out.defined()) {
    TORCH_CHECK(
        out.is_floating_point(),
        fname,
        " expects a floating point output tensor, but got ",
        out.scalar_type());
    auto out_mut = out;
    return _fft_c2r_musa_out(
        input, dim, norm, last_dim_size.guard_int(__FILE__, __LINE__), out_mut);
  }
  return _fft_c2r_musa(
      input, dim, norm, last_dim_size.guard_int(__FILE__, __LINE__));
}

// Real to complex FFT
Tensor fft_r2c_impl(
    c10::string_view function_name,
    Tensor out,
    Tensor input,
    std::optional<SymInt> n_opt,
    int64_t unwrapped_dim,
    std::optional<c10::string_view> norm_str,
    bool forward,
    bool onesided) {
  TORCH_CHECK(
      !input.is_complex(),
      function_name,
      " expects a real input tensor, but got ",
      input.scalar_type());
  TORCH_CHECK(
      !out.defined() || out.is_complex(),
      function_name,
      " expects a complex output tensor, but got ",
      out.scalar_type());
  input = promote_tensor_fft(input);
  const auto input_dim = input.dim();
  const auto dim =
      maybe_wrap_dim(unwrapped_dim, input_dim, /*wrap_scalar=*/false);
  const auto n = n_opt.value_or(input.sym_sizes()[dim]);
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  if (n_opt) {
    input = resize_fft_input(input, dim, n);
  }

  const auto norm = norm_from_string(norm_str, forward);

  Tensor ret;
  if (out.defined() && forward) {
    ret = _fft_r2c_musa_out(
        input, dim, static_cast<int64_t>(norm), onesided, out);
  } else {
    ret = _fft_r2c_musa(input, dim, static_cast<int64_t>(norm), onesided);
  }

  if (!forward) {
    // FIXME: _fft_r2c doesn't support native r2c IFFT
    return out.defined() ? at::conj_physical_out(out, ret) : ret.conj();
  } else {
    return ret;
  }
}

// Complex to real FFT
Tensor fft_c2r_impl(
    c10::string_view function_name,
    Tensor out,
    Tensor input,
    std::optional<SymInt> n_opt,
    int64_t unwrapped_dim,
    std::optional<c10::string_view> norm_str,
    bool forward) {
  TORCH_CHECK(
      !out.defined() || out.is_floating_point(),
      function_name,
      " expects a floating point output tensor, but got ",
      out.scalar_type());
  input = promote_tensor_fft(input, /*require_complex=*/true);
  const auto input_dim = input.dim();
  const auto dim =
      maybe_wrap_dim(unwrapped_dim, input_dim, /*wrap_scalar=*/false);
  const auto n = n_opt.value_or(2 * (input.sym_sizes()[dim] - 1));
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  if (n_opt) {
    input = resize_fft_input(input, dim, n / 2 + 1);
  }
  const auto norm = norm_from_string(norm_str, forward);
  if (forward) {
    // FIXME: _fft does not support complex_output=false with inverse=false
    input = input.conj();
  }
  return fft_c2r_maybe_out(
      function_name, out, input, dim, static_cast<int64_t>(norm), n);
}

// Complex to complex FFT
Tensor fft_c2c_impl(
    c10::string_view function_name,
    Tensor out,
    Tensor input,
    std::optional<SymInt> n_opt,
    int64_t unwrapped_dim,
    std::optional<c10::string_view> norm_str,
    bool forward) {
  TORCH_CHECK(
      input.is_complex(),
      function_name,
      " expects a complex input tensor, but got ",
      input.scalar_type());
  const auto input_dim = input.dim();
  const auto dim =
      maybe_wrap_dim(unwrapped_dim, input_dim, /*wrap_scalar=*/false);
  const auto n = n_opt.value_or(input.sym_sizes()[dim]);
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  if (n_opt) {
    input = resize_fft_input(input, dim, n);
  }
  const auto norm = static_cast<int64_t>(norm_from_string(norm_str, forward));
  return fft_c2c_maybe_out(function_name, out, input, dim, norm, forward);
}

Tensor FftFftMusa_symint(
    const Tensor& self,
    std::optional<SymInt> n,
    int64_t dim,
    std::optional<c10::string_view> norm) {
  return self.is_complex()
      ? fft_c2c_impl("fft", {}, self, n, dim, norm, /*forward=*/true)
      : fft_r2c_impl(
            "fft",
            {},
            self,
            n,
            dim,
            norm,
            /*forward=*/true,
            /*onesided=*/false);
}

Tensor& FftFftMusaOut_symint(
    const Tensor& self,
    std::optional<SymInt> n,
    int64_t dim,
    std::optional<c10::string_view> norm,
    Tensor& out) {
  TORCH_CHECK(
      false,
      "MUSA does not support writing complex tensors (out) for this operation.")
  if (self.is_complex()) {
    fft_c2c_impl("fft", out, self, n, dim, norm, /*forward=*/true);
  } else {
    fft_r2c_impl(
        "fft", out, self, n, dim, norm, /*forward=*/true, /*onesided=*/false);
  }
  return out;
}

Tensor FftIfftMusa_symint(
    const Tensor& self,
    std::optional<SymInt> n,
    int64_t dim,
    std::optional<c10::string_view> norm) {
  return self.is_complex()
      ? fft_c2c_impl("ifft", {}, self, n, dim, norm, /*forward=*/false)
      : fft_r2c_impl(
            "ifft",
            {},
            self,
            n,
            dim,
            norm,
            /*forward=*/false,
            /*onesided=*/false);
}

Tensor& FftIfftMusaOut_symint(
    const Tensor& self,
    std::optional<SymInt> n,
    int64_t dim,
    std::optional<c10::string_view> norm,
    Tensor& out) {
  TORCH_CHECK(
      false,
      "MUSA does not support writing complex tensors (out) for this operation.")
  if (self.is_complex()) {
    fft_c2c_impl("ifft", out, self, n, dim, norm, /*forward=*/false);
  } else {
    fft_r2c_impl(
        "ifft", out, self, n, dim, norm, /*forward=*/false, /*onesided=*/false);
  }
  return out;
}

Tensor FftRfftMusa_symint(
    const Tensor& self,
    std::optional<SymInt> n,
    int64_t dim,
    std::optional<c10::string_view> norm) {
  return fft_r2c_impl(
      "rfft", {}, self, n, dim, norm, /*forward=*/true, /*onesided=*/true);
}

Tensor& FftRfftMusaOut_symint(
    const Tensor& self,
    std::optional<SymInt> n,
    int64_t dim,
    std::optional<c10::string_view> norm,
    Tensor& out) {
  TORCH_CHECK(
      false,
      "MUSA does not support writing complex tensors (out) for this operation.")
  fft_r2c_impl(
      "rfft", out, self, n, dim, norm, /*forward=*/true, /*onesided=*/true);
  return out;
}

Tensor FftIrfftMusa_symint(
    const Tensor& self,
    std::optional<SymInt> n,
    int64_t dim,
    std::optional<c10::string_view> norm) {
  return fft_c2r_impl("irfft", {}, self, n, dim, norm, /*forward=*/false);
}

Tensor& FftIrfftMusaOut_symint(
    const Tensor& self,
    std::optional<SymInt> n,
    int64_t dim,
    std::optional<c10::string_view> norm,
    Tensor& out) {
  TORCH_CHECK(
      false,
      "MUSA does not support writing complex tensors (out) for this operation.")
  fft_c2r_impl("irfft", out, self, n, dim, norm, /*forward=*/false);
  return out;
}

} // namespace musa
} // namespace at