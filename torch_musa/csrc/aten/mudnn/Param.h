#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_PARAM_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_PARAM_H_

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include <mudnn.h>

#include <torch_musa/csrc/aten/mudnn/Exception.h>

namespace at::musa {

struct MudnnParam final {
  mudnnParamDataType_t type = MUDNN_PARAM_DATA_NONE;
  union {
    double d;
    int64_t i;
  } value;

  void Set(int64_t v) noexcept {
    type = MUDNN_PARAM_DATA_INT64;
    value.i = v;
  }

  void Set(double v) noexcept {
    type = MUDNN_PARAM_DATA_DOUBLE;
    value.d = v;
  }

  void* Ptr() {
    if (type == MUDNN_PARAM_DATA_NONE) {
      return nullptr;
    }
    if (type == MUDNN_PARAM_DATA_DOUBLE) {
      return static_cast<void*>(&value.d);
    }
    TORCH_INTERNAL_ASSERT(type == MUDNN_PARAM_DATA_INT64);
    return static_cast<void*>(&value.i);
  }

  mudnnParamDataType_t Type() const noexcept {
    return type;
  }
};

} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_PARAM_H_
