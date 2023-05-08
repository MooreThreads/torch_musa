#pragma once
#include <ATen/musa/MUSAConfig.h>
#include <string>

// AT_USE_JITERATOR(), controls whether we jit some elementwise kernels
#define AT_USE_JITERATOR() false
#define jiterator_stringify(...) std::string(#__VA_ARGS__);
