#ifndef TORCH_MUSA_CSRC_CORE_STREAM_H_
#define TORCH_MUSA_CSRC_CORE_STREAM_H_

#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>

#include "torch_musa/csrc/core/MUSAStream.h"

struct THMPStream : THPStream {
  c10::musa::MUSAStream musa_stream;
};
extern PyObject* THMPStreamClass;

void THMPStream_init(PyObject* module);

inline bool THMPStream_Check(PyObject* obj) {
  return THMPStreamClass && PyObject_IsInstance(obj, THMPStreamClass);
}

#endif // TORCH_MUSA_CSRC_CORE_STREAM_H_
