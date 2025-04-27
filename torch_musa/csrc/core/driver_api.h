#pragma once
#include <musa.h>

#define C10_MUSA_DRIVER_CHECK(EXPR)                                        \
  do {                                                                     \
    MUresult __err = EXPR;                                                 \
    if (__err != MUSA_SUCCESS) {                                           \
      const char* err_str;                                                 \
      MUresult get_error_str_err C10_UNUSED =                              \
          c10::musa::DriverAPI::get()->muGetErrorString_(__err, &err_str); \
      if (get_error_str_err != MUSA_SUCCESS) {                             \
        AT_ERROR("MUSA driver error: unknown error");                      \
      } else {                                                             \
        AT_ERROR("MUSA driver error: ", err_str);                          \
      }                                                                    \
    }                                                                      \
  } while (0)

#define C10_LIBMUSA_DRIVER_API(_) \
  _(muDeviceGetAttribute)         \
  _(muGetErrorString)

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION >= 4000)
#define C10_LIBMUSA_DRIVER_API_4000(_) \
  _(muMemAddressReserve)               \
  _(muMemRelease)                      \
  _(muMemMap)                          \
  _(muMemAddressFree)                  \
  _(muMemSetAccess)                    \
  _(muMemUnmap)                        \
  _(muMemCreate)
#else
#define C10_LIBMUSA_DRIVER_API_4000(_)
#endif

namespace c10 {
namespace musa {

struct DriverAPI {
#define CREATE_MEMBER(name) decltype(&name) name##_;
  C10_LIBMUSA_DRIVER_API(CREATE_MEMBER)
  C10_LIBMUSA_DRIVER_API_4000(CREATE_MEMBER)

#undef CREATE_MEMBER
  static DriverAPI* get();
};

} // namespace musa
} // namespace c10
