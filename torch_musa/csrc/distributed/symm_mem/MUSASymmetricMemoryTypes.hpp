#pragma once

#include <cstdint>

#include <musa.h>

namespace c10d::symmetric_memory {

// Covers MTL8
constexpr int max_musa_p2p_domain_size = 8;
// Maximum number of channels
constexpr int symm_max_nblocks = 32;

// Maximally, a rank will need to sync with all other ranks, over all
// channels. Each signal is 32 bits, which is the minimum unit for atomic cas.
constexpr size_t signal_pad_size =
    symm_max_nblocks * max_musa_p2p_domain_size * sizeof(uint32_t);


using HandleType = MUmemGenericAllocationHandle;
} // namespace c10d::symmetric_memory
