#ifndef TORCH_MUSA_CSRC_CORE_MUSAMISCFUNCTIONS_H_
#define TORCH_MUSA_CSRC_CORE_MUSAMISCFUNCTIONS_H_
// this file is to avoid circular dependency between MUSAFunctions.h and
// MUSAExceptions.h

#include <mutex>

namespace c10 {
namespace musa {
const char* get_musa_check_suffix() noexcept;
std::mutex* getFreeMutex();
} // namespace musa
} // namespace c10

#endif // TORCH_MUSA_CSRC_CORE_MUSAMISCFUNCTIONS_H_
