#ifndef TORCH_MUSA_CSRC_ATEN_UTILS_TENSORITERATOR_H_
#define TORCH_MUSA_CSRC_ATEN_UTILS_TENSORITERATOR_H_

#include <functional>

#include <ATen/TensorIterator.h>

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
namespace musa {

class MusaTensorIterator : public TensorIteratorBase {
 public:
  void musa_promote_inputs_to_common_dtype(bool flag) noexcept {
    do_promote_inputs_to_common_dtype_ = flag;
  }

  void set_musa_common_dtype_lifter(
      std::function<ScalarType(ScalarType)> lifter) noexcept {
    common_dtype_lifter_ = std::move(lifter);
  }

  void build(TensorIteratorConfig&);

  void cast_outputs();

  void add_owned_output(const TensorBase& output);

  void add_borrowed_output(const TensorBase& output);

  void add_owned_input(const TensorBase& input);

  void add_borrowed_input(const TensorBase& input);

  virtual void add_output(const TensorBase& output) {
    add_borrowed_output(output);
  }

  void add_input(const TensorBase& input) {
    add_borrowed_input(input);
  }

  void replace_input(int arg, const TensorBase& input);

  muTensor mu_input(int arg) const;

  muTensor mu_output(int arg) const;

  bool input_is_type_corrected(int arg) const;

  bool output_is_type_corrected(int arg) const;

 protected:
  virtual void _set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options);

  virtual void mark_inplace();

  optional<ScalarType> common_dtype_for_functional_output() {
    if (cast_common_dtype_to_outputs_) {
      return promote_common_dtype_;
    }
    return c10::nullopt;
  }

 private:
  void check_set_device(at::Device device);

  void check_propagate_names(int64_t output_idx, DimnameList names);

  const Tensor& maybe_get_output(int64_t output_idx) override;

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override;

  StrideVector compatible_stride() const;

  muTensor mu_tensor(int arg) const;

  bool tensor_is_type_corrected(int arg) const;

  void compute_types(const TensorIteratorConfig&);

  FastSetupType compute_fast_setup_type(const TensorIteratorConfig&);

  bool fast_set_up(const TensorIteratorConfig&);

  void compute_strides(const TensorIteratorConfig&);

  void reorder_dimensions();

  void allocate_or_resize_outputs();

 private:
  bool do_reorder_dimensions_ = false;
  bool do_promote_inputs_to_common_dtype_ = true;
  bool cast_common_dtype_to_outputs_ = false;
  ScalarType promote_common_dtype_ = ScalarType::Undefined;
  std::function<ScalarType(ScalarType)> common_dtype_lifter_;
  c10::musa::OptionalMUSAGuard guard_;
};

class FunctionalTensorIterator : public MusaTensorIterator {
 public:
  void add_output(const TensorBase& output) override {
    add_owned_output(output);
  }

 private:
  void _set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options) override;

  void mark_inplace() override {}
};

class InplaceTensorIterator : public MusaTensorIterator {
 private:
  void _set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options) override;

  void mark_inplace() override;
};

class OutTensorIterator : public MusaTensorIterator {
 private:
  void _set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options) override;
};

} // namespace musa
} // namespace at

#endif // TORCH_MUSA_CSRC_ATEN_UTILS_TENSORITERATOR_H_
