#include "torch_musa/csrc/aten/utils/TensorIterator.h"

#include <ATen/NamedTensorUtils.h>
#include <ATen/native/Resize.h>
#include <c10/util/DimVector.h>
#include <c10/util/Exception.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"

namespace at {
namespace musa {

using StrideVector = MusaTensorIterator::StrideVector;

namespace {

Tensor create_out(
    IntArrayRef sizes,
    IntArrayRef strides,
    TensorOptions options) {
  return strides.empty()
      ? at::detail::empty_musa(sizes, options)
      : at::detail::empty_strided_musa(sizes, strides, options);
}

void restride_out(
    const Tensor& out,
    IntArrayRef sizes,
    IntArrayRef strides,
    TensorOptions options) {
  const auto opt_memory_format = options.memory_format_opt();
  const bool has_memory_format = opt_memory_format.has_value();
  if (!strides.empty()) {
    TORCH_INTERNAL_ASSERT(!has_memory_format);
    out.as_strided_(sizes, strides);
  } else if (has_memory_format) {
    out.unsafeGetTensorImpl()->empty_tensor_restride(*opt_memory_format);
  }
}

void resize_out(
    const Tensor& out,
    IntArrayRef sizes,
    IntArrayRef strides,
    TensorOptions options) {
  at::native::resize_output(out, sizes);
  restride_out(out, sizes, strides, options);
}

void create_output_raw_strided_no_check(
    OperandInfo& op,
    IntArrayRef sizes,
    IntArrayRef strides,
    TensorOptions options,
    optional<ScalarType> backup_dtype) {
  op.tensor(
      c10::MaybeOwned<TensorBase>::owned(create_out(sizes, strides, options)));
  if (backup_dtype.has_value()) {
    const auto c_type = (*backup_dtype);
    if (c_type != op.target_dtype) {
      op.exchange_tensor(c10::MaybeOwned<TensorBase>::owned(
          create_out(sizes, strides, options.dtype(c_type))));
      op.target_dtype = c_type;
    }
  }
  op.current_dtype = op.target_dtype;
}

void resize_output_raw_strided_no_check(
    const OperandInfo& op,
    IntArrayRef sizes,
    IntArrayRef strides,
    TensorOptions options) {
  resize_out(op.tensor(), sizes, strides, options);
  const auto& original_t = op.original_tensor();
  if (original_t.defined()) {
    resize_out(original_t, sizes, strides, options);
  }
}

TensorOptions original_options(const OperandInfo& op) {
  if (op.original_tensor_base().defined()) {
    return op.original_tensor_base().options();
  } else {
    return op.options();
  }
}

const Tensor& original_output_tensor(const OperandInfo& op) {
  if (op.original_tensor_base().defined()) {
    return op.original_tensor();
  }
  return op.tensor();
}

} // anonymous namespace

void MusaTensorIterator::check_set_device(at::Device device) {
  const auto current_device = guard_.current_device();
  if (!current_device.has_value()) {
    guard_.reset_device(device);
    return;
  }
  TORCH_INTERNAL_ASSERT(
      (*current_device) == device,
      "musa structured kernels don't support multi-device outputs");
}

void MusaTensorIterator::check_propagate_names(
    int64_t output_idx,
    DimnameList names) {
  if (!names.empty()) {
    namedinference::propagate_names(
        original_output_tensor(operands_[output_idx]), names);
  }
}

const Tensor& MusaTensorIterator::maybe_get_output(int64_t output_idx) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(output_idx < num_outputs_);
  return original_output_tensor(operands_[output_idx]);
}

void MusaTensorIterator::_set_output_raw_strided(
    int64_t output_idx,
    IntArrayRef sizes,
    IntArrayRef strides,
    TensorOptions options) {
  auto& op = operands_[output_idx];
  const auto& base = op.tensor_base();
  if (!base.defined()) {
    create_output_raw_strided_no_check(
        op, sizes, strides, options, common_dtype_for_functional_output());
  } else if (op.will_resize) {
    resize_output_raw_strided_no_check(op, sizes, strides, options);
  }
}

void MusaTensorIterator::set_output_raw_strided(
    int64_t output_idx,
    IntArrayRef sizes,
    IntArrayRef strides,
    TensorOptions options,
    DimnameList names) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(output_idx < num_outputs_);
  check_set_device(options.device());
  _set_output_raw_strided(output_idx, sizes, strides, options);
  check_propagate_names(output_idx, names);
}

StrideVector MusaTensorIterator::compatible_stride() const {
  int dim = ndim();
  auto stride = StrideVector(dim, 0);
  int64_t next_stride = 1;
  --dim;
  for (; dim >= 0; --dim) {
    stride[dim] = next_stride;
    next_stride *= shape_[dim];
  }
  return stride;
}

void MusaTensorIterator::add_owned_output(const TensorBase& output) {
  TORCH_INTERNAL_ASSERT(
      static_cast<size_t>(num_outputs_) == operands_.size(),
      "Outputs must be added before any inputs.");
  operands_.emplace_back(
      c10::MaybeOwned<TensorBase>::owned(std::in_place, output));
  operands_[num_outputs_].is_output = true;
  ++num_outputs_;
}

void MusaTensorIterator::add_borrowed_output(const TensorBase& output) {
  TORCH_INTERNAL_ASSERT(
      static_cast<size_t>(num_outputs_) == operands_.size(),
      "Outputs must be added before any inputs.");
  operands_.emplace_back(c10::MaybeOwned<TensorBase>::borrowed(output));
  operands_[num_outputs_].is_output = true;
  ++num_outputs_;
}

void MusaTensorIterator::add_owned_input(const TensorBase& input) {
  TORCH_INTERNAL_ASSERT(num_outputs_ > 0, "At least one output must be added.");
  operands_.emplace_back(
      c10::MaybeOwned<TensorBase>::owned(std::in_place, input));
}

void MusaTensorIterator::add_borrowed_input(const TensorBase& input) {
  TORCH_INTERNAL_ASSERT(num_outputs_ > 0, "At least one output must be added.");
  operands_.emplace_back(c10::MaybeOwned<TensorBase>::borrowed(input));
}

void MusaTensorIterator::replace_input(int arg, const TensorBase& input) {
  arg += num_outputs_;
  AT_ASSERT(arg >= num_outputs_ && arg < ntensors());
  auto& op = operands_[arg];
  const auto& old_input = op.tensor();
  const auto new_sizes = input.sizes();
  const auto new_strides = input.strides();
  const auto new_dtype = input.scalar_type();
  AT_ASSERT(old_input.sizes() == new_sizes);
  if (old_input.strides() != new_strides) {
    AT_ASSERT(!old_input.is_non_overlapping_and_dense());
    const int dims = ndim();
    const int offset = dims - static_cast<int>(new_sizes.size());
    for (const auto i : c10::irange(dims)) {
      auto& stride = op.stride_bytes[i];
      if (stride != 0) {
        stride = new_strides[perm_[i] - offset];
      }
    }
  }
  op.tensor(c10::MaybeOwned<TensorBase>::owned(std::in_place, input));
  op.data = input.data_ptr();
  op.current_dtype = new_dtype;
  op.target_dtype = new_dtype;
}

muTensor MusaTensorIterator::mu_input(int arg) const {
  arg += num_outputs_;
  AT_ASSERT(arg >= num_outputs_ && arg < ntensors());
  return mu_tensor(arg);
}

muTensor MusaTensorIterator::mu_output(int arg) const {
  AT_ASSERT(arg >= 0 && arg < num_outputs_);
  return mu_tensor(arg);
}

muTensor MusaTensorIterator::mu_tensor(int arg) const {
  muTensor mt;
  auto& op = operands_[arg];
  SetMUTensorDType(op.current_dtype, mt);
  SetMUTensorAddr(op.data, mt);

  mt.SetNdInfo(ndim(), shape_.data(), op.stride_bytes.data());
  return mt;
}

const Tensor& MusaTensorIterator::original_input(int arg) const {
  arg += num_outputs_;
  AT_ASSERT(arg >= num_outputs_ && arg < ntensors());
  return original_output_tensor(operands_[arg]);
}

bool MusaTensorIterator::input_is_type_corrected(int arg) const {
  arg += num_outputs_;
  AT_ASSERT(arg >= num_outputs_ && arg < ntensors());
  return tensor_is_type_corrected(arg);
}

bool MusaTensorIterator::output_is_type_corrected(int arg) const {
  AT_ASSERT(arg >= 0 && arg < num_outputs_);
  return tensor_is_type_corrected(arg);
}

bool MusaTensorIterator::tensor_is_type_corrected(int arg) const {
  const auto& op = operands_[arg];
  return !original_output_tensor(op).is_same(op.tensor());
}

void MusaTensorIterator::mark_inplace() {
  for (const auto i : c10::irange(num_outputs_)) {
    auto& op = operands_[i];
    const auto& output = op.tensor_base();
    if (!output.defined()) {
      continue;
    }

    for (const auto j : c10::irange(num_outputs_, ntensors())) {
      const auto& input = tensor_base(j);
      if (output.is_same(input)) {
        op.is_read_write = true;
        break;
      }
    }
  }
}

void MusaTensorIterator::compute_types(const TensorIteratorConfig& config) {
  TensorIteratorBase::compute_types(config);

  promote_common_dtype_ = common_dtype_lifter_
      ? common_dtype_lifter_(common_dtype_)
      : common_dtype_;

  for (const auto i : c10::irange(ntensors())) {
    auto& op = operands_[i];
    const auto& base = op.tensor_base();
    if (!base.defined()) {
      continue;
    }

    if (config.cast_common_dtype_to_outputs_ && op.is_output &&
        op.current_dtype != promote_common_dtype_) {
      TORCH_INTERNAL_ASSERT(base.defined());
      const auto opt = base.options().dtype(promote_common_dtype_);
      if (op.will_resize) {
        op.exchange_tensor(c10::MaybeOwned<TensorBase>::owned(
            at::detail::empty_musa({0}, opt)));
      } else if (base.is_non_overlapping_and_dense()) {
        op.exchange_tensor(c10::MaybeOwned<TensorBase>::owned(
            at::detail::empty_strided_musa(base.sizes(), base.strides(), opt)));
      } else {
        op.exchange_tensor(c10::MaybeOwned<TensorBase>::owned(
            at::empty_like(op.tensor(), opt, at::MemoryFormat::Contiguous)));
      }
      op.current_dtype = promote_common_dtype_;
      op.target_dtype = promote_common_dtype_;
    }

    if (config.promote_inputs_to_common_dtype_ &&
        do_promote_inputs_to_common_dtype_ && !op.is_output &&
        op.current_dtype != promote_common_dtype_ && !is_cpu_scalar(i)) {
      op.exchange_tensor(c10::MaybeOwned<TensorBase>::owned(
          op.tensor().to(promote_common_dtype_)));
      op.current_dtype = promote_common_dtype_;
      op.target_dtype = promote_common_dtype_;
    }
  }
}

FastSetupType MusaTensorIterator::compute_fast_setup_type(
    const TensorIteratorConfig& config) {
  if (is_reduction_ || !all_ops_same_shape_) {
    return FastSetupType::NONE;
  }

  // For linear iteration, only contiguous tensors can be coalesced
  // Fast setup of any other format requires changing iteration order
  if (C10_UNLIKELY(enforce_linear_iteration_)) {
    for (const auto& op : operands_) {
      const auto& base = op.tensor_base();
      if (base.defined() && !op.will_resize) {
        if (!base.is_contiguous(at::MemoryFormat::Contiguous)) {
          return FastSetupType::NONE;
        }
      }
    }
    return FastSetupType::CONTIGUOUS;
  }

  bool is_contig = true;
  bool is_channels_last_contig = true, is_channels_last_like = false;
  bool is_channels_last3d_contig = true, is_channels_last3d_like = false;
  bool is_non_overlapping_and_dense = true;

  for (const auto& op : operands_) {
    const auto& base = op.tensor_base();
    if (base.defined() && !op.will_resize) {
      is_contig &= base.is_contiguous(at::MemoryFormat::Contiguous);

      is_channels_last_contig &=
          base.is_contiguous(at::MemoryFormat::ChannelsLast);
      is_channels_last_like |=
          (base.suggest_memory_format() == at::MemoryFormat::ChannelsLast);

      is_channels_last3d_contig &=
          base.is_contiguous(at::MemoryFormat::ChannelsLast3d);
      is_channels_last3d_like |=
          (base.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d);

      is_non_overlapping_and_dense &= base.is_non_overlapping_and_dense();
    }
  }

  if (is_contig) {
    if (is_channels_last_contig && is_channels_last_like) {
      return FastSetupType::CHANNELS_LAST;
    }
    if (is_channels_last3d_contig && is_channels_last3d_like) {
      return FastSetupType::CHANNELS_LAST_3D;
    }
    return FastSetupType::CONTIGUOUS;
  }
  if (is_channels_last_contig) {
    return FastSetupType::CHANNELS_LAST;
  }
  if (is_channels_last3d_contig) {
    return FastSetupType::CHANNELS_LAST_3D;
  }
  if (is_non_overlapping_and_dense) {
    int prev = -1;
    for (int i = ntensors() - 1; i >= 0; --i) {
      const auto& op = operands_[i];
      const auto& op_base = op.tensor_base();
      if (op_base.defined() && !op.will_resize) {
        if (prev < 0) {
          prev = i;
          continue;
        }
        if (!tensor_base(prev).strides().equals(op_base.strides())) {
          return FastSetupType::NONE;
        }
      }
    }
    return FastSetupType::NON_OVERLAPPING_DENSE;
  }
  return FastSetupType::NONE;
}

bool MusaTensorIterator::fast_set_up(const TensorIteratorConfig& config) {
  const auto setup_type = compute_fast_setup_type(config);
  if (setup_type == FastSetupType::NONE) {
    return false;
  }

  switch (setup_type) {
    case FastSetupType::CONTIGUOUS: {
      for (const auto i : c10::irange(num_outputs_)) {
        auto& op = operands_[i];
        if (!op.tensor_base().defined()) {
          TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
        }
        set_output_raw_strided(
            i,
            shape_,
            {},
            original_options(op).memory_format(MemoryFormat::Contiguous),
            names_);
      }
      break;
    }
    case FastSetupType::CHANNELS_LAST: {
      for (const auto i : c10::irange(num_outputs_)) {
        auto& op = operands_[i];
        if (!op.tensor_base().defined()) {
          TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
        }
        set_output_raw_strided(
            i,
            shape_,
            {},
            original_options(op).memory_format(MemoryFormat::ChannelsLast),
            names_);
      }
      break;
    }
    case FastSetupType::CHANNELS_LAST_3D: {
      for (const auto i : c10::irange(num_outputs_)) {
        auto& op = operands_[i];
        if (!op.tensor_base().defined()) {
          TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
        }
        set_output_raw_strided(
            i,
            shape_,
            {},
            original_options(op).memory_format(MemoryFormat::ChannelsLast3d),
            names_);
      }
      break;
    }
    case FastSetupType::NON_OVERLAPPING_DENSE: {
      int i_defined = ntensors() - 1;
      for (; i_defined >= 0 && !tensor(i_defined).defined(); --i_defined) {
      }
      TORCH_CHECK(
          i_defined >= 0,
          "Can not find a defined tensor when fast allocating memory to outputs");
      for (const auto i : c10::irange(num_outputs_)) {
        auto& op = operands_[i];
        if (!op.tensor_base().defined()) {
          TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
        }
        set_output_raw_strided(
            i,
            shape_,
            tensor_base(i_defined).strides(),
            original_options(op),
            names_);
      }
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unsupported fast setup type",
          c10::to_string(static_cast<int>(setup_type)));
  }

  if (ndim() > 1) {
    has_coalesced_dimensions_ = true;
  }
  const auto n_elems = numel();
  shape_.resize(1UL);
  shape_[0] = n_elems;
  for (auto& op : operands_) {
    op.stride_bytes.resize(1UL, 0);
    if (op.tensor().dim() > 0) {
      op.stride_bytes[0] = 1;
    }
  }
  return true;
}

void MusaTensorIterator::compute_strides(const TensorIteratorConfig& config) {
  for (auto& op : operands_) {
    const auto& op_base = op.tensor_base();
    if (op_base.defined() && !op.will_resize) {
      const IntArrayRef original_shape =
          config.static_shape_ ? shape_ : op_base.sizes();
      const auto original_stride = op_base.strides();
      const auto original_dims = static_cast<int>(original_shape.size());

      const auto broadcasted_dims = ndim();
      const auto offset = broadcasted_dims - original_dims;
      op.stride_bytes.resize(broadcasted_dims, 0);

      for (const auto i : c10::irange(original_dims)) {
        const auto offset_i = offset + i;
        if (original_shape[i] == 1 && shape_[offset_i] != 1) {
          op.stride_bytes[offset_i] = 0;
        } else {
          op.stride_bytes[offset_i] = original_stride[i];
        }
      }
    }
  }
}

void MusaTensorIterator::reorder_dimensions() {
  const auto broadcasted_dims = ndim();
  perm_.resize(broadcasted_dims);
  std::iota(perm_.begin(), perm_.end(), 0);

  if (enforce_linear_iteration_) {
    return;
  }

  auto should_swap = [&](size_t dim0, size_t dim1) {
    for (const auto arg : c10::irange(ntensors())) {
      if (operands_[arg].stride_bytes.empty() || operands_[arg].will_resize) {
        continue;
      }
      int64_t stride0 = operands_[arg].stride_bytes[dim0];
      int64_t stride1 = operands_[arg].stride_bytes[dim1];
      if (is_reduction_ && operands_[arg].is_output) {
        if ((stride0 == 0) != (stride1 == 0)) {
          return stride1 == 0 ? 1 : -1;
        }
      }

      if (stride0 == 0 || stride1 == 0) {
        continue;
      } else if (stride0 < stride1) {
        return -1;
      } else if (stride0 > stride1) {
        return 1;
      } else {
        auto t_dim0 = shape_[dim0];
        auto t_dim1 = shape_[dim1];
        if (t_dim0 > t_dim1) {
          return 1;
        }
      }
    }
    return 0;
  };

  for (auto i = broadcasted_dims - 2; i >= 0; --i) {
    int dim1 = i;
    for (auto dim0 = i + 1; dim0 < broadcasted_dims; ++dim0) {
      int comparison = should_swap(perm_[dim0], perm_[dim1]);
      if (comparison > 0) {
        std::swap(perm_[dim0], perm_[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }

  for (const auto i : c10::irange(broadcasted_dims)) {
    if (perm_[i] != i) {
      do_reorder_dimensions_ = true;
      break;
    }
  }
  if (do_reorder_dimensions_) {
    permute_dimensions(perm_);
  }
}

void MusaTensorIterator::allocate_or_resize_outputs() {
  for (const auto i : c10::irange(num_outputs_)) {
    auto& op = operands_[i];
    const auto& op_base = op.tensor_base();
    const auto opt = original_options(op);
    if (!op_base.defined() || op.will_resize) {
      TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
      op.stride_bytes = compatible_stride();
      if (!do_reorder_dimensions_) {
        set_output_raw_strided(i, shape_, {}, opt, names_);
      } else {
        const auto tensor_shape = invert_perm(shape_);
        const auto tensor_stride = invert_perm(op.stride_bytes);
        set_output_raw_strided(i, tensor_shape, tensor_stride, opt, names_);
      }
      op.current_dtype = op.target_dtype;
    } else if (op_base.defined()) {
      set_output_raw_strided(i, op_base.sizes(), {}, opt, names_);
    }
  }
}

void MusaTensorIterator::build(TensorIteratorConfig& config) {
  is_reduction_ = config.is_reduction_;
  enforce_linear_iteration_ = config.enforce_linear_iteration_;
  cast_common_dtype_to_outputs_ = config.cast_common_dtype_to_outputs_;

  mark_inplace();

  compute_mem_overlaps(config);

  compute_names(config);

  compute_shape(config);

  mark_resize_outputs(config);

  compute_types(config);

  if (!fast_set_up(config)) {
    compute_strides(config);

    reorder_dimensions();

    allocate_or_resize_outputs();
  }

  common_dtype_ = promote_common_dtype_;

  for (auto& op : operands_) {
    const auto& op_base = op.tensor_base();
    TORCH_INTERNAL_ASSERT(op_base.defined());
    op.data = op_base.data_ptr();
  }
}

void MusaTensorIterator::cast_outputs() {
  for (const auto i : c10::irange(num_outputs_)) {
    auto& op = operands_[i];
    const auto& original_t = op.original_tensor();
    if (original_t.defined() && original_t.scalar_type() != op.current_dtype) {
      original_t.copy_(op.tensor());
      op.restore_original_tensor();
    }
  }
}

void FunctionalTensorIterator::_set_output_raw_strided(
    int64_t output_idx,
    IntArrayRef sizes,
    IntArrayRef strides,
    TensorOptions options) {
  auto& op = operands_[output_idx];
  TORCH_CHECK(
      !op.tensor_base().defined(),
      "musa functional structured kernels don't support defined outputs");
  create_output_raw_strided_no_check(
      op, sizes, strides, options, common_dtype_for_functional_output());
}

void InplaceTensorIterator::_set_output_raw_strided(
    int64_t output_idx,
    IntArrayRef sizes,
    IntArrayRef strides,
    TensorOptions options) {
  auto& op = operands_[output_idx];
  TORCH_CHECK(
      op.is_read_write,
      "musa inplace structured kernels don't support not readable/writable outputs");
}

void InplaceTensorIterator::mark_inplace() {
  for (const auto i : c10::irange(num_outputs_)) {
    operands_[i].is_read_write = true;
  }
}

void OutTensorIterator::_set_output_raw_strided(
    int64_t output_idx,
    IntArrayRef sizes,
    IntArrayRef strides,
    TensorOptions options) {
  auto& op = operands_[output_idx];
  if (op.will_resize) {
    resize_output_raw_strided_no_check(op, sizes, strides, options);
  }
}

} // namespace musa
} // namespace at
