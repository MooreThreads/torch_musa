cmake_minimum_required(VERSION 3.10)

project(op_subproject LANGUAGES CXX)

FILE(GLOB _CSRCS
    aten/ops/*.cpp
    aten/utils/*.cpp
    aten/mudnn/*.cpp
    core/*.cpp
    utils/*.cpp
    distributed/*.cpp
    aten/quantized/*.cpp
)

LIST(APPEND MUSA_CSRCS ${_CSRCS})

# Pass to parent
set(MUSA_CSRCS ${MUSA_CSRCS} PARENT_SCOPE)

list(APPEND CMAKE_MODULE_PATH $ENV{MUSA_HOME}/cmake)
find_package(MUSA REQUIRED)

FILE(GLOB ATEN_MUSA_CSRCS aten/musa/*.cpp)

FILE(GLOB PORTING_CPP_CSRCS
    aten/ops/musa/*.cpp
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Activation.cpp)

FILE(GLOB MU_SRCS
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UpSampleBicubic2d.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Repeat.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UpSampleBilinear2d.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReflectionPad.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnarySignKernels.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/PointwiseOpsKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/BinaryMiscBackwardOpsKernels.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/BinaryMiscOpsKernels.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnaryOpsKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/TensorCompare.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/BinaryBitwiseOpsKernels.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/group_norm_kernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/FillKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/WeightNorm.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/IndexKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/LegacyThrustHelpers.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/musa/cub.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/musa/detail/IndexUtils.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/EmbeddingBackwardKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/TriangularOps.mu

    # Compiling following files takes too long, disable them temporarily
    # ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Embedding.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/musa/cub-RadixSortPairs.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReduceNormKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReduceMaxValuesKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/TensorTransformations.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationHardswishKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationHardsigmoidKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationLogSigmoidKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationGeluKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/CompareKernels.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/LossCTC.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionUniform.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionNormal.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionBernoulli.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationPreluKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationHardtanhKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationSoftplusKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DilatedMaxPool3d.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionExponentialKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/TriangularOps.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/RangeFactories.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Indexing.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationGluKernel.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/GridSampler.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/quantized/musa/IntReprQuant.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/quantized/musa/MakePerTensorQuantizedTensor.mu
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/NLLLoss2d.mu
    aten/quantized/musa/*.mu
    aten/ops/musa/Embedding.mu
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

# TODO(@caizhi): Use FindOpenMP.cmake to replace absolute paths
musa_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/aten/ops/musa /usr/lib/llvm-11/include/openmp/)
musa_add_library(${MUSA_KERNELS_LIB} SHARED ${MU_SRCS} ${PORTING_CPP_CSRCS} ${ATEN_MUSA_CSRCS})

set(CUBLAS_LIB $ENV{MUSA_HOME}/lib/libmublas.so)
target_link_libraries(${MUSA_KERNELS_LIB} ${CUBLAS_LIB})

if(NOT MUSA_KERNELS_INSTALL_LIB_DIR)
    set(MUSA_KERNELS_INSTALL_LIB_DIR lib)
endif()

install(TARGETS ${MUSA_KERNELS_LIB} DESTINATION ${MUSA_KERNELS_INSTALL_LIB_DIR})
