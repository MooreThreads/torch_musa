#ifndef USE_MUSA
#error "test_aoti_musa_shim.cpp requires USE_MUSA"
#endif

#ifndef AOT_INDUCTOR_USE_CACHING_ALLOCATOR
#error "RAII_gpuMalloc test requires AOT_INDUCTOR_USE_CACHING_ALLOCATOR"
#endif

#include <torch/csrc/inductor/aoti_runtime/model_base.h>
#include <torch/csrc/inductor/aoti_runtime/utils_musa.h>
#include <torch/csrc/stable/library.h>

#include <cstdint>
#include <stdexcept>

namespace {

int32_t current_device_index() {
  int32_t device_index = -1;
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_device_index(&device_index));
  return device_index;
}

void expect_current_device_index(int32_t expected) {
  const int32_t actual = current_device_index();
  if (actual != expected) {
    throw std::runtime_error("current MUSA device index did not match");
  }
}

void* current_musa_stream(int32_t device_index) {
  void* stream = nullptr;
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_musa_stream(device_index, &stream));
  return stream;
}

void expect_current_musa_stream(void* expected, int32_t device_index) {
  void* actual = current_musa_stream(device_index);
  if (actual != expected) {
    throw std::runtime_error("current MUSA stream did not match");
  }
}

} // namespace

void my_aoti_create_musa_guard(int32_t device_index) {
  const int32_t previous_device_index = current_device_index();

  MUSAGuardHandle guard = nullptr;
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_create_musa_guard(device_index, &guard));
  expect_current_device_index(device_index);

  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_musa_guard_set_index(guard, device_index));
  expect_current_device_index(device_index);

  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_musa_guard(guard));
  expect_current_device_index(previous_device_index);
}

void my_aoti_create_musa_stream_guard(void* stream, int32_t device_index) {
  void* previous_stream = current_musa_stream(device_index);

  MUSAStreamGuardHandle guard = nullptr;
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_create_musa_stream_guard(stream, device_index, &guard));
  expect_current_musa_stream(stream, device_index);

  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_musa_stream_guard(guard));
  expect_current_musa_stream(previous_stream, device_index);
}

bool my_aoti_musa_caching_allocator_alloc_delete(int64_t nbytes) {
  if (nbytes < 0) {
    throw std::runtime_error("nbytes must be non-negative");
  }

  void* ptr = nullptr;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_musa_caching_allocator_raw_alloc(
      static_cast<uint64_t>(nbytes), &ptr));
  const bool allocated = ptr != nullptr;
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_musa_caching_allocator_raw_delete(ptr));
  return allocated;
}

bool my_aoti_model_base_raii_gpu_malloc(int64_t nbytes) {
  if (nbytes < 0) {
    throw std::runtime_error("nbytes must be non-negative");
  }

  auto ptr = RAII_gpuMalloc(static_cast<size_t>(nbytes));
  return ptr.get() != nullptr;
}

void my_aoti_musa_runtime_raii_guard(void* stream, int32_t device_index) {
  const int32_t previous_device_index = current_device_index();
  void* previous_stream = current_musa_stream(device_index);

  {
    torch::aot_inductor::AOTIMusaGuard guard(device_index);
    guard.set_index(device_index);
    expect_current_device_index(device_index);

    torch::aot_inductor::AOTIMusaStreamGuard stream_guard(
        static_cast<musaStream_t>(stream), device_index);
    expect_current_musa_stream(stream, device_index);
  }

  expect_current_device_index(previous_device_index);
  expect_current_musa_stream(previous_stream, device_index);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_aoti_create_musa_guard(int device_index) -> ()");
  m.def("my_aoti_create_musa_stream_guard(int stream, int device_index) -> ()");
  m.def("my_aoti_musa_caching_allocator_alloc_delete(int nbytes) -> bool");
  m.def("my_aoti_model_base_raii_gpu_malloc(int nbytes) -> bool");
  m.def("my_aoti_musa_runtime_raii_guard(int stream, int device_index) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_aoti_create_musa_guard", TORCH_BOX(&my_aoti_create_musa_guard));
  m.impl(
      "my_aoti_create_musa_stream_guard",
      TORCH_BOX(&my_aoti_create_musa_stream_guard));
  m.impl(
      "my_aoti_musa_caching_allocator_alloc_delete",
      TORCH_BOX(&my_aoti_musa_caching_allocator_alloc_delete));
  m.impl(
      "my_aoti_model_base_raii_gpu_malloc",
      TORCH_BOX(&my_aoti_model_base_raii_gpu_malloc));
  m.impl(
      "my_aoti_musa_runtime_raii_guard",
      TORCH_BOX(&my_aoti_musa_runtime_raii_guard));
}
