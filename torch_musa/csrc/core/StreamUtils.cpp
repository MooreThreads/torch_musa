#include <torch/csrc/musa/Stream.h>
#include <torch/csrc/musa/THCP.h>
#include <torch/csrc/python_headers.h>

#include <cstdarg>
#include <string>

namespace torch {
namespace musa {
// NB: It's a list of *optional* MUSAStream; when nullopt, that means to use
// whatever the current stream of the device the input is associated with was.
std::vector<c10::optional<c10::musa::MUSAStream>>
THPUtils_PySequence_to_MUSAStreamList(PyObject* obj) {
  if (!PySequence_Check(obj)) {
    throw std::runtime_error(
        "Expected a sequence in THPUtils_PySequence_to_MUSAStreamList");
  }
  THPObjectPtr seq = THPObjectPtr(PySequence_Fast(obj, nullptr));
  if (seq.get() == nullptr) {
    throw std::runtime_error(
        "expected PySequence, but got " + std::string(THPUtils_typename(obj)));
  }

  std::vector<c10::optional<c10::musa::MUSAStream>> streams;
  Py_ssize_t length = PySequence_Fast_GET_SIZE(seq.get());
  for (Py_ssize_t i = 0; i < length; i++) {
    PyObject* stream = PySequence_Fast_GET_ITEM(seq.get(), i);

    if (PyObject_IsInstance(stream, THCPStreamClass)) {
      // Spicy hot reinterpret cast!!
      streams.emplace_back(c10::musa::MUSAStream::unpack3(
          (reinterpret_cast<THCPStream*>(stream))->stream_id,
          (reinterpret_cast<THCPStream*>(stream))->device_index,
          static_cast<c10::DeviceType>(
              (reinterpret_cast<THCPStream*>(stream))->device_type)));
    } else if (stream == Py_None) {
      streams.emplace_back();
    } else {
      // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
      std::runtime_error(
          "Unknown data type found in stream list. Need torch.musa.Stream or None");
    }
  }
  return streams;
}
} // namespace musa
} // namespace torch