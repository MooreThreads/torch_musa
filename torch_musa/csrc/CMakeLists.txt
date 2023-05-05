cmake_minimum_required(VERSION 3.10)

project(op_subproject LANGUAGES CXX)

FILE(GLOB _CSRCS
    aten/ops/*.cpp
    aten/utils/*.cpp
    aten/mudnn/*.cpp
    core/*.cpp
    )

LIST(APPEND MUSA_CSRCS ${_CSRCS})

# Pass to parent
set(MUSA_CSRCS ${MUSA_CSRCS} PARENT_SCOPE)


list(APPEND CMAKE_MODULE_PATH $ENV{MUSA_HOME}/cmake)
find_package(MUSA REQUIRED)

FILE(GLOB aten_musa_cpp aten/musa/*.cpp)

FILE(GLOB cuda_porting_cpp_files
    aten/ops/musa/*.cpp
    # TODO(MTAI): following cpp files should be implemented inside torch_musa
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/c10/musa/MUSA_PORT_CachingAllocator.cpp
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/c10/musa/MUSA_PORT_MallocAsyncAllocator.cpp
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/c10/musa/MUSA_PORT_MiscFunctions.cpp)

FILE(GLOB cuda_porting_mu_files
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/group_norm_kernel.mu
    core/*.mu
    )

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/utils.cmake)
append_cxx_flag_if_supported("-Wno-unused-parameter" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-unused-variable" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-sign-compare" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-w" CMAKE_CXX_FLAGS)
set(MUSA_MCC_FLAGS -D__MUSA_NO_HALF_OPERATORS__)
set(MUSA_MCC_FLAGS -U__CUDA__)

if(NOT ENABLE_COMPILE_FP64)
    string(APPEND MUSA_MCC_FLAGS " -mathx-disable-fp64")  # disable fp64
endif()
#set(MUSA_VERBOSE_BUILD ON)
musa_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/aten/ops/musa)
musa_add_library(${MUSA_KERNELS_LIB} SHARED ${cuda_porting_mu_files} ${cuda_porting_cpp_files} ${aten_musa_cpp})

set(CUBLAS_LIB $ENV{MUSA_HOME}/lib/libmublas.so)
target_link_libraries(${MUSA_KERNELS_LIB} ${CUBLAS_LIB})

if(NOT MUSA_KERNELS_INSTALL_LIB_DIR)
  set(MUSA_KERNELS_INSTALL_LIB_DIR lib)
endif()

install(TARGETS ${MUSA_KERNELS_LIB} DESTINATION ${MUSA_KERNELS_INSTALL_LIB_DIR})
