#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_DESCRIPTORS_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_DESCRIPTORS_H_

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include <memory>

#include <mudnn.h>

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

template <typename T, mudnnStatus_t (*dtor)(T*)>
struct DescriptorDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      AT_MUDNN_CHECK(dtor(x));
    }
  }
};

template <typename T, mudnnStatus_t (*ctor)(T**), mudnnStatus_t (*dtor)(T*)>
class Descriptor {
 public:
  Descriptor() {
    Init();
  }

  T* Desc() const noexcept {
    return desc_.get();
  }
  T* Desc() noexcept {
    return desc_.get();
  }

 protected:
  void Init() {
    T* raw_desc = nullptr;
    AT_MUDNN_CHECK(ctor(&raw_desc));
    desc_.reset(raw_desc);
  }

 private:
  std::unique_ptr<T, DescriptorDeleter<T, dtor>> desc_;
};

} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API
#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_DESCRIPTORS_H_
