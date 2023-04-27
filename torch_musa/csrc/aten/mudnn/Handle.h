#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_HANDLE_H
#define TORCH_MUSA_CSRC_ATEN_MUDNN_HANDLE_H
#include "mudnn.h"

namespace at {
namespace native {

using mudnnHandle_t = ::musa::dnn::Handle*;

::musa::dnn::Handle& getMudnnHandle();
} // namespace native
} // namespace at

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_HANDLE_H
