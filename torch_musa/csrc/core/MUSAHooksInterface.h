#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

namespace at {

struct MUSAHooksInterface {
  virtual ~MUSAHooksInterface() = default;

  virtual void initMUSA() const {
    TORCH_CHECK(false, "Cannot initialize MUSA without torch_musa library.")
  }

  virtual c10::Device getDeviceFromPtr(void* /*data*/) const {
    TORCH_CHECK(false, "Cannot initialize MUSA without torch_musa library.")
  }

  virtual bool hasMUSA() const {
    return false;
  }

  virtual int64_t current_device() const {
    return -1;
  }

  virtual int getNumGPUs() const {
    return 0;
  }

  virtual void deviceSynchronize(int64_t /*device_index*/) const {
    TORCH_CHECK(false, "Cannot initialize MUSA without torch_musa library.")
  }
};

struct MUSAHooksArgs {};

C10_DECLARE_REGISTRY(MUSAHooksRegistry, MUSAHooksInterface, MUSAHooksArgs);
#define REGISTER_MUSA_HOOKS(clsname) \
  C10_REGISTER_CLASS(MUSAHooksRegistry, clsname, clsname)

namespace detail {
const MUSAHooksInterface& getMUSAHooks();
} // namespace detail
} // namespace at
