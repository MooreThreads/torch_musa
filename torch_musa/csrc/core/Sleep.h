#ifndef TORCH_MUSA_CSRC_CORE_SLEEP_H_
#define TORCH_MUSA_CSRC_CORE_SLEEP_H_

#include <cstdint>

namespace at {
namespace musa {

// enqueues a kernel that spins for the specified number of cycles
void sleep(int64_t cycles);

} // namespace musa
} // namespace at

#endif // TORCH_MUSA_CSRC_CORE_SLEEP_H_
