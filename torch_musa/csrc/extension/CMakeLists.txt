cmake_minimum_required(VERSION 3.10)

file(GLOB _EXT_MU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/apex/*.mu)
musa_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/apex)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/utils.cmake)
append_cxx_flag_if_supported("-Wno-unused-parameter" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-unused-variable" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-sign-compare" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-w" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-std=c++17" CMAKE_CXX_FLAGS)
string(APPEND MUSA_MCC_FLAGS " -U__CUDA__")

if(NOT ENABLE_COMPILE_FP64)
  string(APPEND MUSA_MCC_FLAGS " -mathx-disable-fp64") # disable fp64
endif()

musa_add_library(${EXT_MUSA_KERNEL_LIB} SHARED ${_EXT_MU_SRCS})

if(NOT EXT_MUSA_KERNEL_INSTALL_LIB_DIR)
  set(EXT_MUSA_KERNEL_INSTALL_LIB_DIR lib)
endif()
install(TARGETS ${EXT_MUSA_KERNEL_LIB}
        DESTINATION ${EXT_MUSA_KERNEL_INSTALL_LIB_DIR})
