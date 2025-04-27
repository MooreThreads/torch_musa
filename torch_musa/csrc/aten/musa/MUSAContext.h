#ifndef TORCH_MUSA_CSRC_ATEN_MUSA_MUSACONTEXT_H_
#define TORCH_MUSA_CSRC_ATEN_MUSA_MUSACONTEXT_H_

#include <cstdint>

#include <mublas.h>
#include <musa_runtime_api.h>
#include <musparse.h>

#include <ATen/Context.h>
#include <ATen/core/ATenGeneral.h>

#include "torch_musa/csrc/aten/musa/Exceptions.h"
#include "torch_musa/csrc/core/MUSAFunctions.h"
#include "torch_musa/csrc/core/MUSAHooksInterface.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace musa {

/*
A common MUSA interface for ATen.
This interface is distinct from MUSAHooks, which defines an interface that links
to both CPU-only and MUSA builds. That interface is intended for runtime
dispatch and should be used from files that are included in both CPU-only and
MUSA builds.
MUSAContext, on the other hand, should be preferred by files only included in
MUSA builds. It is intended to expose MUSA functionality in a consistent
manner.
This means there is some overlap between the MUSAContext and MUSAHooks, but
the choice of which to use is simple: use MUSAContext when in a MUSA-only file,
use MUSAHooks otherwise.
Note that MUSAContext simply defines an interface with no associated class.
It is expected that the modules whose functions compose this interface will
manage their own state. There is only a single MUSA context/state.
*/

using c10::musa::memcpy_and_sync;

/**
 * DEPRECATED: use device_count() instead
 */
inline int64_t getNumGPUs() {
  return c10::musa::device_count();
}

/**
 * MUSA is available if we compiled with MUSA, and there are one or more
 * devices.  If we compiled with MUSA but there is a driver problem, etc.,
 * this function will report MUSA is not available (rather than raise an error.)
 */
inline bool is_available() {
  return c10::musa::device_count() > 0;
}

musaDeviceProp* getCurrentDeviceProperties();

musaDeviceProp* getDeviceProperties(int device);

bool canDeviceAccessPeer(int device, int peer_device);

int warp_size();

Allocator* getMUSADeviceAllocator();

mublasHandle_t getCurrentMUSABlasHandle();

inline void lazyInitMUSA() {
  static c10::once_flag thm_init;
  c10::call_once(thm_init, [&] { at::detail::getMUSAHooks().initMUSA(); });
}

uint32_t getMUSAArch();

uint32_t getMUSAArch(int device);

bool maybeDNNOpSupportBFloat16();

bool maybeDNNOpSupportBFloat16(int device);

} // namespace musa
} // namespace at
#endif // TORCH_MUSA_CSRC_ATEN_MUSA_MUSACONTEXT_H_
