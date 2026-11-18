#pragma once
// Packaged under the upstream PyTorch directory layout for downstream users.
#include "torch_musa/csrc/core/MUSAEvent.h"

namespace c10::musa {
namespace detail {
using at::musa::detail::EventPoolBase;
}

using at::musa::MUSAEvent;
}