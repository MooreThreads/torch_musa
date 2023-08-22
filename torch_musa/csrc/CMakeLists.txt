cmake_minimum_required(VERSION 3.10)

project(op_subproject LANGUAGES CXX)

FILE(GLOB _CSRCS
    aten/ops/*.cpp
    aten/utils/*.cpp
    aten/mudnn/*.cpp
    core/*.cpp
    amp/*.cpp
    utils/*.cpp
    aten/quantized/*.cpp
    aten/quantized/mudnn/*.cpp
)

FILE(GLOB _MUDNN_ATTENTION_CSRCS aten/ops/attention/mudnn/*.cpp)
LIST(APPEND MUSA_CSRCS ${_CSRCS} ${_MUDNN_ATTENTION_CSRCS})

if(USE_MCCL)
  FILE(GLOB _MCCL_CSRCS distributed/*.cpp)
  list(APPEND MUSA_CSRCS ${_MCCL_CSRCS})
endif()

LIST(APPEND MUSA_CSRCS ${_CSRCS})

# Pass to parent
set(MUSA_CSRCS ${MUSA_CSRCS} PARENT_SCOPE)

list(APPEND CMAKE_MODULE_PATH $ENV{MUSA_HOME}/cmake)
find_package(MUSA REQUIRED)

FILE(GLOB ATEN_MUSA_CSRCS 
    aten/musa/*.cpp
)

FILE(GLOB PORTING_CPP_CSRCS
    aten/ops/musa/*.cpp
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Activation.cpp)

FILE(GLOB MU_SRCS
    ${GENERATED_PORTING_DIR}/aten/src/ATen/musa/cub.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/musa/cub-RadixSortKeys.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/musa/cub-RadixSortPairs.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/musa/detail/IndexUtils.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationHardswishKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationHardsigmoidKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationLogSigmoidKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationGeluKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationPreluKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationHardtanhKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationSoftplusKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationGluKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/AmpKernels.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/AveragePool3d.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/BinaryMiscBackwardOpsKernels.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/BinaryMiscOpsKernels.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/BinaryBitwiseOpsKernels.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/CompareKernels.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/CrossKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionUniform.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionNormal.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionBernoulli.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionRandomKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DilatedMaxPool3d.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionExponentialKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Embedding.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/EmbeddingBackwardKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/FillKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/group_norm_kernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/GridSampler.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Indexing.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/IndexKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/LegacyThrustHelpers.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/LossCTC.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/MultinomialKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/NLLLoss2d.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/PointwiseOpsKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/RangeFactories.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Repeat.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReflectionPad.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReduceNormKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReduceMaxValuesKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReduceAMinMaxKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Randperm.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Shape.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/TensorCompare.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/TensorTransformations.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/TriangularOps.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnaryFractionKernels.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnarySignKernels.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Unique.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UniqueCub.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UpSampleBicubic2d.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UpSampleBilinear2d.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UpSampleLinear1d.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UpSampleNearest3d.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnaryOpsKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/WeightNorm.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReplicationPadding.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UpSampleNearest1d.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/quantized/musa/Activation.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/quantized/musa/IntReprQuant.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/quantized/musa/MakePerTensorQuantizedTensor.mu
    aten/quantized/musa/*.mu
    aten/ops/torchvision/*.mu
    aten/ops/musa/*.mu
    core/*.mu
)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/utils.cmake)
append_cxx_flag_if_supported("-Wno-unused-parameter" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-unused-variable" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-sign-compare" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-w" CMAKE_CXX_FLAGS)
string(APPEND MUSA_MCC_FLAGS " -U__CUDA__")

if(NOT ENABLE_COMPILE_FP64)
    string(APPEND MUSA_MCC_FLAGS " -mathx-disable-fp64") # disable fp64
endif()

# set openmp include directory
set(llvm_openmp_search_list)
foreach(item RANGE 6 20 1)
  list(APPEND llvm_openmp_search_list /usr/lib/llvm-${item}/include/openmp/)
endforeach()

find_path(
  OPENMP_INCLUDE_DIR omp.h
  PATHS ${llvm_openmp_search_list}
  REQUIRED
  NO_DEFAULT_PATH)

musa_include_directories(${OPENMP_INCLUDE_DIR})
musa_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/aten/ops/musa aten/ops/torchvision)

musa_add_library(${MUSA_KERNELS_LIB} SHARED ${MU_SRCS} ${PORTING_CPP_CSRCS} ${ATEN_MUSA_CSRCS})

set(MUBLAS_LIB $ENV{MUSA_HOME}/lib/libmublas.so)
target_link_libraries(${MUSA_KERNELS_LIB} ${MUBLAS_LIB})

if(NOT MUSA_KERNELS_INSTALL_LIB_DIR)
    set(MUSA_KERNELS_INSTALL_LIB_DIR lib)
endif()

install(TARGETS ${MUSA_KERNELS_LIB} DESTINATION ${MUSA_KERNELS_INSTALL_LIB_DIR})
