cmake_minimum_required(VERSION 3.10)

project(op_subproject LANGUAGES CXX)

file(
  GLOB
  _CSRCS
  amp/*.cpp
  aten/ops/*.cpp
  aten/ops/attention/mudnn/*.cpp
  aten/utils/*.cpp
  aten/mudnn/*.cpp
  aten/quantized/*.cpp
  aten/quantized/mudnn/*.cpp
  core/*.cpp
  profiler/*.cpp
  utils/*.cpp)

list(APPEND MUSA_CSRCS ${_CSRCS} ${GENERATED_MUSA_CSRCS})

if(USE_MCCL)
  file(GLOB _MCCL_CSRCS distributed/*.cpp)
  list(APPEND MUSA_CSRCS ${_MCCL_CSRCS})
endif()

file(GLOB_RECURSE AOTI_MUSA_SRCS inductor/aoti_runner/*.cpp
     inductor/aoti_torch/*.cpp)
list(APPEND MUSA_CSRCS ${AOTI_MUSA_SRCS})

# Pass to parent
set(MUSA_CSRCS ${MUSA_CSRCS} PARENT_SCOPE)

list(APPEND CMAKE_MODULE_PATH $ENV{MUSA_HOME}/cmake)
find_package(MUSA REQUIRED)

file(GLOB ATEN_MUSA_CSRCS aten/musa/*.cpp)

file(
  GLOB
  PORTING_CPP_CSRCS
  aten/ops/musa/*.cpp
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Activation.cpp
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Resize.cpp
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Sorting.cpp
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/TensorModeKernel.cpp
  ${GENERATED_PORTING_DIR}/aten/src/ATen/musa/EmptyTensor.cpp
  ${GENERATED_PORTING_DIR}/torch/csrc/musa/comm.cpp
  ${GENERATED_PORTING_DIR}/torch/csrc/musa/Stream.cpp
  ${GENERATED_PORTING_DIR}/torch/csrc/musa/Distributions.cpp
  ${GENERATED_PORTING_DIR}/torch/csrc/musa/Loss.cpp)

# cmake-lint: disable=C0301
file(
  GLOB_RECURSE
  MU_SRCS
  ${GENERATED_PORTING_DIR}/aten/src/ATen/musa/cub.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/musa/cub-RadixSortKeys.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/musa/cub-RadixSortPairs.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/musa/detail/IndexUtils.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/AbsKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationEluKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationSoftshrinkKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationHardsigmoidKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationHardswishKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationHardshrinkKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationHardtanhKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationLeakyReluKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationLogSigmoidKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationGeluKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationGluKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationMishKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationPreluKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationSoftplusKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ActivationThresholdKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/AdaptiveMaxPooling3d.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/AmpKernels.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/AveragePool3d.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/BinaryMiscBackwardOpsKernels.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/BinaryGeometricKernels.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/BinaryMiscOpsKernels.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/BinaryBitwiseOpsKernels.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/BinaryShiftOpsKernels.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Col2Im.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ComplexKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/CrossKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/CumminmaxKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/CopysignKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionUniform.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionNormal.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionBernoulli.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionCauchyKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionLogNormalKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionGeometricKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionRandomKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DilatedMaxPool3d.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistributionExponentialKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/FractionalMaxPool2d.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/FractionalMaxPool3d.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ForeachBinaryOpList.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ForeachBinaryOpScalar.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ForeachBinaryOpScalarTensor.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ForeachBinaryOpScalarList.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ForeachPointwiseOp.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ForeachTernaryOp.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ForeachUnaryOp.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/group_norm_kernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/GridSampler.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/GcdLcmKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Im2Col.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/IGammaKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/LegacyThrustHelpers.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Lerp.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/LinearAlgebra.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Loss.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/MultinomialKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Nonzero.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Normalization.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/NLLLoss2d.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/PointwiseOpsKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Randperm.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/RangeFactories.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReduceAMinMaxKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReflectionPad.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Repeat.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ReplicationPadding.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/RNN.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/ScatterGatherKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Sorting.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/SoftMax.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/TensorTransformations.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/TriangularOps.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnaryFractionKernels.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnarySignKernels.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnarySpecialOpsKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnaryGammaKernels.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnaryGeometricAsinKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnaryGeometricAsinhKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnaryGeometricSinhKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnaryGeometricAcoshKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnaryGeometricAtanhKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnaryGeometricCoshKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnaryLogKernels.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnfoldBackwardKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UpSampleBicubic2d.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UpSampleTrilinear3d.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UpSampleLinear1d.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UpSampleNearest1d.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UpSampleNearest2d.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UpSampleNearest3d.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/UnaryOpsKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/WeightNorm.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/DistanceKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/LogAddExpKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/StepKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/RenormKernel.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/MaxUnpooling.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/TensorCompare.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/quantized/musa/Activation.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/quantized/musa/IntReprQuant.mu
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/quantized/musa/MakePerTensorQuantizedTensor.mu
  aten/quantized/musa/*.mu
  aten/ops/torchvision/*.mu
  aten/ops/musa/*.mu
  aten/ops/attention/mudnn/*.mu
  core/*.mu)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/utils.cmake)
append_cxx_flag_if_supported("-Wno-unused-parameter" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-unused-variable" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-sign-compare" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-w" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-std=c++17" CMAKE_CXX_FLAGS)
string(APPEND MUSA_MCC_FLAGS " -U__CUDA__")
string(APPEND MUSA_MCC_FLAGS " -mllvm -mtgpu-enable-const-calc")
string(APPEND MUSA_MCC_FLAGS " -mllvm -mtgpu-if-convert")
string(APPEND MUSA_MCC_FLAGS " -mllvm -enable-post-misched")
# string(APPEND MUSA_MCC_FLAGS " -mllvm
# -mtgpu-enable-max-ilp-scheduling-strategy=1")
string(APPEND MUSA_MCC_FLAGS " -mllvm -mtgpu-load-store-opt")

string(APPEND MUSA_MCC_FLAGS " -mllvm -enable-ldma-index")

if(NOT ENABLE_COMPILE_FP64)
  string(APPEND MUSA_MCC_FLAGS " -mathx-disable-fp64") # disable fp64
endif()

find_package(OpenMP)
if(NOT OpenMP_FOUND)
  message(FATAL_ERROR "OpenMP is not found by the cmake in system.")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # do nothing
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # OpenMP_CXX_INCLUDE_DIRS will not be populated with GCC as the compiler. We
  # should extract the include dir from OpenMP_CXX_LIBRARIES
  list(GET OpenMP_CXX_LIBRARIES 0 openmp_library_path)
  get_filename_component(gcc_dir ${openmp_library_path} DIRECTORY)
  set(OpenMP_CXX_INCLUDE_DIRS "${gcc_dir}/include")
else()
  message(
    FATAL_ERROR "[${CMAKE_CXX_COMPILER_ID}] compiler is not supported now.")
endif()

musa_include_directories(${OpenMP_CXX_INCLUDE_DIRS})
musa_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/aten/ops/musa
                         aten/ops/torchvision)

musa_add_library(
  ${MUSA_KERNELS_LIB}
  SHARED
  ${MU_SRCS}
  ${PORTING_CPP_CSRCS}
  ${ATEN_MUSA_CSRCS})

musa_add_mublas_to_target(${MUSA_KERNELS_LIB})

if(NOT MUSA_KERNELS_INSTALL_LIB_DIR)
  set(MUSA_KERNELS_INSTALL_LIB_DIR lib)
endif()

install(TARGETS ${MUSA_KERNELS_LIB} DESTINATION ${MUSA_KERNELS_INSTALL_LIB_DIR})
