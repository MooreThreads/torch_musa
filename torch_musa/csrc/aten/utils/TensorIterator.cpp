#include "torch_musa/csrc/aten/utils/TensorIterator.h"

#include <ATen/NamedTensorUtils.h>
#include <ATen/native/Resize.h>
#include <c10/util/DimVector.h>
#include <c10/util/Exception.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/MudnnUtils.h"

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
      const auto old_elem_size =
          static_cast<int64_t>(c10::elementSize(op.target_dtype));
      op.exchange_tensor(c10::MaybeOwned<TensorBase>::owned(
          create_out(sizes, strides, options.dtype(c_type))));
      op.target_dtype = c_type;
      const auto elem_size =
          static_cast<int64_t>(op.tensor_base().element_size());
      for (auto& byte_stride : op.stride_bytes) {
        TORCH_INTERNAL_ASSERT(byte_stride % old_elem_size == 0);
        byte_stride = byte_stride / old_elem_size * elem_size;
      }
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

DimVector invert_perm_with(IntArrayRef input, IntArrayRef perm) {
  TORCH_INTERNAL_ASSERT(input.size() == perm.size());
  auto inverted = DimVector(input.size());
  for (const auto i : c10::irange(input.size())) {
    inverted[perm[i]] = input[i];
  }
  return inverted;
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
  int dim = element_ndim();
  auto stride = StrideVector(dim, 0);
  int64_t next_stride = 1;
  --dim;
  for (; dim >= 0; --dim) {
    stride[dim] = next_stride;
    next_stride *= element_shape_[dim];
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
    const int dims = element_ndim();
    const int offset = dims - static_cast<int>(new_sizes.size());
    for (const auto i : c10::irange(dims)) {
      auto& stride = op.element_strides[i];
      if (stride != 0) {
        stride = new_strides[element_perm_[i] - offset];
      }
    }
    for (const auto i : c10::irange(ndim())) {
      auto& stride = op.stride_bytes[i];
      if (stride != 0) {
        stride = new_strides[perm_[i] - offset] *
            static_cast<int64_t>(input.element_size());
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

  mt.SetNdInfo(
      element_ndim(), element_shape_.data(), op.element_strides.data());
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

bool MusaTensorIterator::_is_cpu_scalar(int64_t arg) const {
  auto& op = operands_[arg];
  return op.tensor_base().dim() == 0 && op.tensor_base().is_cpu();
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
        op.current_dtype != promote_common_dtype_ && !_is_cpu_scalar(i)) {
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
  element_shape_ = shape_;
  perm_.resize(1UL);
  perm_[0] = 0;
  element_perm_.resize(1UL);
  element_perm_[0] = 0;
  do_reorder_element_dimensions_ = false;
  for (auto& op : operands_) {
    op.element_strides.resize(1UL, 0);
    op.stride_bytes.resize(1UL, 0);
    if (op.tensor().dim() > 0) {
      op.element_strides[0] = 1;
      op.stride_bytes[0] =
          static_cast<int64_t>(op.tensor_base().element_size());
    }
  }
  return true;
}

void MusaTensorIterator::compute_element_shape_and_strides() {
  const auto dims = ndim();
  element_shape_.resize(dims);
  element_perm_.resize(dims);
  do_reorder_element_dimensions_ = false;
  for (const auto i : c10::irange(dims)) {
    const auto dim = dims - 1 - i;
    element_shape_[i] = shape_[dim];
    // element_shape_/element_strides are already reversed from the base
    // iterator order. element_perm_ is only for MUSA output allocation, which
    // this path no longer uses after base allocation and coalescing.
    element_perm_[i] = i;
  }

  for (const auto arg : c10::irange(ntensors())) {
    auto& op = operands_[arg];
    const auto& op_base = op.tensor_base();
    op.element_strides.resize(dims);
    const auto elem_size = static_cast<int64_t>(op_base.element_size());
    for (const auto i : c10::irange(dims)) {
      const auto byte_stride = op.stride_bytes[dims - 1 - i];
      TORCH_INTERNAL_ASSERT(
          byte_stride % elem_size == 0,
          "byte stride must be divisible by element size");
      op.element_strides[i] = byte_stride / elem_size;
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

    TensorIteratorBase::allocate_or_resize_outputs();

    if (!is_meta_) {
      TensorIteratorBase::coalesce_dimensions();
    }

    compute_element_shape_and_strides();
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
