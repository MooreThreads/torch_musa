#ifndef TORCH_MUSA_CSRC_ATEN_UTILS_TENSORITERATOR_H_
#define TORCH_MUSA_CSRC_ATEN_UTILS_TENSORITERATOR_H_

#include <ATen/TensorIterator.h>

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
namespace musa {

class MusaTensorIterator : public TensorIteratorBase {
 public:
  void musa_promote_inputs_to_common_dtype(bool flag) {
    do_promote_inputs_to_common_dtype_ = flag;
  }

  void build(TensorIteratorConfig&);

  void cast_outputs() {
    _cast_outputs();
  }

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

  virtual bool output_is_type_corrected(int arg) const;

 protected:
  virtual void _set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options);

  virtual void _cast_outputs();

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

  virtual void mark_inplace();

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
  c10::musa::OptionalMUSAGuard guard_;
};

class FunctionalTensorIterator : public MusaTensorIterator {
 public:
  void add_output(const TensorBase& output) override {
    add_owned_output(output);
  }

  bool output_is_type_corrected(int arg) const override {
    return false;
  }

 private:
  void _set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options) override;

  void mark_inplace() override {}

  void _cast_outputs() override {}
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
