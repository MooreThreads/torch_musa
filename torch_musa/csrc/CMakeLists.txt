cmake_minimum_required(VERSION 3.10)

project(op_subproject LANGUAGES CXX)

FILE(GLOB _CSRCS
    aten/ops/*.cpp
    aten/utils/*.cpp
    aten/mudnn/*.cpp
    core/*.cpp
    utils/*.cpp
    )

LIST(APPEND MUSA_CSRCS ${_CSRCS})

# Pass to parent
set(MUSA_CSRCS ${MUSA_CSRCS} PARENT_SCOPE)


list(APPEND CMAKE_MODULE_PATH $ENV{MUSA_HOME}/cmake)
find_package(MUSA REQUIRED)

FILE(GLOB ATEN_MUSA_CSRCS aten/musa/*.cpp)

FILE(GLOB PORTING_CSRCS
    aten/ops/musa/*.cpp)

FILE(GLOB MU_SRCS
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReflectionPad.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnarySignKernels.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/PointwiseOpsKernel.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/BinaryMiscBackwardOpsKernels.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/BinaryMiscOpsKernels.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnaryOpsKernel.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/TensorCompare.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/BinaryBitwiseOpsKernels.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/group_norm_kernel.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/FillKernel.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/WeightNorm.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/IndexKernel.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/LegacyThrustHelpers.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/musa/cub.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/EmbeddingBackwardKernel.mu
    # Compiling following files takes too long, disable them temporarily
    # ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Embedding.mu
    # ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Indexing.mu
    # ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/musa/cub-RadixSortPairs.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReduceNormKernel.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReduceMaxValuesKernel.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReflectionPad.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/TensorTransformations.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationHardswishKernel.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationHardsigmoidKernel.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionUniform.mu
    ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UpSampleBilinear2d.mu
    core/*.mu
    )

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/utils.cmake)
append_cxx_flag_if_supported("-Wno-unused-parameter" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-unused-variable" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-sign-compare" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-w" CMAKE_CXX_FLAGS)
string(APPEND MUSA_MCC_FLAGS " -U__CUDA__")

if(NOT ENABLE_COMPILE_FP64)
    string(APPEND MUSA_MCC_FLAGS " -mathx-disable-fp64")  # disable fp64
endif()
#set(MUSA_VERBOSE_BUILD ON)
# TODO(@caizhi): Use FindOpenMP.cmake to replace absolute paths
musa_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/aten/ops/musa /usr/lib/llvm-11/include/openmp/)
musa_add_library(${MUSA_KERNELS_LIB} SHARED ${MU_SRCS} ${PORTING_CSRCS} ${ATEN_MUSA_CSRCS})

set(CUBLAS_LIB $ENV{MUSA_HOME}/lib/libmublas.so)
target_link_libraries(${MUSA_KERNELS_LIB} ${CUBLAS_LIB})

if(NOT MUSA_KERNELS_INSTALL_LIB_DIR)
  set(MUSA_KERNELS_INSTALL_LIB_DIR lib)
endif()

install(TARGETS ${MUSA_KERNELS_LIB} DESTINATION ${MUSA_KERNELS_INSTALL_LIB_DIR})
