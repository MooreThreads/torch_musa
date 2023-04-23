#ifndef TORCH_MUSA_CSRC_CORE_THPSTREAM_INC
#define TORCH_MUSA_CSRC_CORE_THPSTREAM_INC

#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>

#include "torch_musa/csrc/core/MUSAStream.h"

struct THMPStream : THPStream {
  torch_musa::MUSAStream musa_stream;
};
extern PyObject* THMPStreamClass;

void THMPStream_init(PyObject* module);

inline bool THMPStream_Check(PyObject* obj) {
  return THMPStreamClass && PyObject_IsInstance(obj, THMPStreamClass);
}

#endif // TORCH_MUSA_CSRC_CORE_THPSTREAM_INC
