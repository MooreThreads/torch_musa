#pragma once
// Packaged under the upstream PyTorch directory layout for downstream users.
#include "torch_musa/csrc/core/PeerToPeerAccess.h"

namespace c10::musa {
namespace detail {
using at::musa::detail::init_p2p_access_cache;
}

using at::musa::get_p2p_access;
}
