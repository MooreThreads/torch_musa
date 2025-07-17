#ifndef TORCH_MUSA_CSRC_CORE_PEERTOPEERACCESS_H_
#define TORCH_MUSA_CSRC_CORE_PEERTOPEERACCESS_H_

#include <cstdint>

namespace at::musa {

namespace detail {

void init_p2p_access_cache(int64_t num_devices);

}

bool get_p2p_access(int source_dev, int dest_dev);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_CORE_PEERTOPEERACCESS_H_
