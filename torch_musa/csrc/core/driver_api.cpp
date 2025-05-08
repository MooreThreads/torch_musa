#include "torch_musa/csrc/core/driver_api.h"

#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>

namespace c10::musa {

namespace {

DriverAPI create_driver_api() {
  void* handle_0 = dlopen("libmusa.so", RTLD_LAZY | RTLD_NOLOAD);
  TORCH_CHECK(handle_0, "Can't open libmusa.so: ", dlerror());
  DriverAPI r{};

#define LOOKUP_LIBMUSA_ENTRY(name)                       \
  r.name##_ = ((decltype(&name))dlsym(handle_0, #name)); \
  TORCH_INTERNAL_ASSERT(r.name##_, "Can't find ", #name, ": ", dlerror())
  C10_LIBMUSA_DRIVER_API(LOOKUP_LIBMUSA_ENTRY)
  C10_LIBMUSA_DRIVER_API_4000(LOOKUP_LIBMUSA_ENTRY)
#undef LOOKUP_LIBMUSA_ENTRY

  return r;
}

} // anonymous namespace

DriverAPI* DriverAPI::get() {
  static DriverAPI singleton = create_driver_api();
  return &singleton;
}

} // namespace c10::musa
