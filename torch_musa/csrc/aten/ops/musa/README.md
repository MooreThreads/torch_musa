This document describes how the op kernel is implemented through cuda porting.

### How to Porting CUDA Kernels for an Operation
0. Investigate how the op is called in 'build/aten/src/ATen/RegisterCUDA.cpp' which is generated when building PyTorch. For example,
```cpp
// abs
at::Tensor & wrapper_CUDA_out_abs_out(const at::Tensor & self, at::Tensor & out) {
    // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::abs_out(self, out);
}
...
m.impl("abs.out", TORCH_FN(wrapper_CUDA_out_abs_out));
...
```
```cpp
// glu
at::Tensor wrapper_CUDA_glu(const at::Tensor & self, int64_t dim) {
  // No device check
structured_glu_out_functional op;
op.meta(self, dim);
op.impl(self, dim, *op.outputs_[0]);
return std::move(op.outputs_[0]).take();
}
...
m.impl("glu", TORCH_FN(wrapper_CUDA_glu));
...
```
1. Add the corresponding cu/cuh/cpp/h file to torch_musa/csrc/CMakeLists.txt. For example,
```cpp
FILE(GLOB cuda_porting_cu_files
    ${CMAKE_BINARY_DIR}/generated_cuda_compatible/aten/src/ATen/native/musa/AbsKernel.cu)
```
2. Link registration and implementation. For example,
```cpp
#include <torch/library.h>
#include <ATen/ops/abs_native.h>
at::Tensor& MusaAbsout(const at::Tensor & self, at::Tensor & out) {
  return at::native::abs_out(self, out);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("abs.out", &MusaAbsout);
}
```
```cpp
#include <build/generated_cuda_compatible/aten/src/ATen/ops/glu_native.h>

extern Tensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options);

struct structured_glu_out_functional final : public at::native::structured_glu_out {
    void set_output_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
        outputs_[output_idx] = ::at::native::musa::create_out(sizes, strides, options);
        //if (!names.empty()) {
        //  namedinference::propagate_names(*outputs_[output_idx], names);
        //}
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_glu_out::set_output_raw_strided(output_idx, sizes, strides, options, names);
    }
    void set_output_raw_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
        outputs_[output_idx] = ::at::native::musa::create_out(sizes, strides, options);
        //if (!names.empty()) {
        //  namedinference::propagate_names(*outputs_[output_idx], names);
        //}
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_glu_out::set_output_raw_strided(output_idx, sizes, strides, options, names);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
      return *outputs_[output_idx];
    }
    std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
};

at::Tensor Glu(const at::Tensor & self, int64_t dim) {
  structured_glu_out_functional op;
  op.meta(self, dim);
  op.impl(self, dim, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("glu", &Glu);
}
```
3. The key to making this system work is that we replaced the macro 'REGISTER_CUDA_DISPATCH' with 'REGISTER_MUSA_DISPATCH' in 'build/generated_cuda_compatible/aten/src/ATen/native/DispatchStub.h'. For more details please refer to 'DispatchStub.h' and 'torch_musa/tools/cuda_porting/README.md'.
