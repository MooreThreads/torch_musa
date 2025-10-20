#ifndef TORCH_MUSA_CSRC_CORE_MCCL_H_
#define TORCH_MUSA_CSRC_CORE_MCCL_H_

#include <cstdint>

namespace torch::musa::mccl {

std::uint64_t version();

const char* version_suffix();

} // namespace torch::musa::mccl

#endif // TORCH_MUSA_CSRC_CORE_MCCL_H_
