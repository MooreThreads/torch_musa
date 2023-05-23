#include "torch_musa/csrc/core/MUSAHooksInterface.h"

namespace at {
namespace musa {
namespace detail {

struct MUSAHooks : public MUSAHooksInterface {
  MUSAHooks(MUSAHooksArgs) {}
  void initMUSA() const override;
  Device getDeviceFromPtr(void* data) const override;
  bool hasMUSA() const override;
  int64_t current_device() const override;
  int getNumGPUs() const override;
  void deviceSynchronize(int64_t device_index) const override;
};

} // namespace detail
} // namespace musa
} // namespace at
