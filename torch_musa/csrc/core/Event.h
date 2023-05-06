#ifndef THMP_EVENT_INC
#define THMP_EVENT_INC

#include <torch/csrc/python_headers.h>

#include "torch_musa/csrc/core/MUSAEvent.h"

struct THMPEvent {
  PyObject_HEAD at::musa::MUSAEvent musa_event;
};
extern PyObject* THMPEventClass;

void THMPEvent_init(PyObject* module);

inline bool THMPEvent_Check(PyObject* obj) {
  return THMPEventClass && PyObject_IsInstance(obj, THMPEventClass);
}

#endif // THMP_EVENT_INC

