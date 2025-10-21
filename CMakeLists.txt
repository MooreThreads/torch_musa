cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(Torch_MUSA CXX C)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(LINUX TRUE)
else()
  message(FATAL_ERROR "torch_musa could be built only on Linux now!")
endif()

set(CMAKE_INSTALL_MESSAGE NEVER)

if(NOT DEFINED BUILD_PYTORCH_REPO_PATH)
  message(FATAL_ERROR "torch_musa needs BUILD_PYTORCH_REPO_PATH"
                      "set to find some modules in PyTorch repo.")
endif()

# We should include PyTorch customized cmake modules into our search path.
list(APPEND CMAKE_MODULE_PATH ${BUILD_PYTORCH_REPO_PATH}/cmake/Modules)

# CMake Build Options according to PyTorch. This cmake block are dedicated for
# these options.
include(cmake/cpu_flags.cmake)
option(USE_PER_OPERATOR_HEADERS
       "Whether ATen should generate separate headers for each operator" ON)

# Build Options above shoud keep consistent with PyTorch

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)

if(env_cxx_standard GREATER -1)
  message(
    WARNING "C++ standard version definition detected in environment variable."
            "PyTorch requires -std=c++17."
            "Please remove -std=c++ settings in your environment.")
endif()

set(CMAKE_CXX_STANDARD 17
    CACHE STRING
          "The C++ standard whose features are requested to build this target.")
set(CMAKE_C_STANDARD 11
    CACHE STRING
          "The C standard whose features are requested to build this target.")

if(DEFINED GLIBCXX_USE_CXX11_ABI)
  if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
    set(CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Googletest's cmake files are going to set it on once they are processed. Let's
# set it at the very beginning so that the entire build is deterministic.
set(THREADS_PREFER_PTHREAD_FLAG ON)

option(ENABLE_COMPILE_FP64 "Enable FP64" OFF)
option(USE_ASAN "Use Address Sanitizer" OFF)
option(USE_TSAN "Use Thread Sanitizer" OFF)
option(USE_MUSA "Use MUSA" ON)
option(USE_MCCL "USE MCCL" ON)
option(MUSA_VERBOSE_BUILD "Print Building details of musa compiling unit." OFF)
include(CMakeDependentOption)

cmake_dependent_option(
  USE_MUDNN
  "Use muDNN"
  ON
  "USE_MUSA"
  OFF)

cmake_dependent_option(
  USE_CCACHE
  "Attempt using CCache to wrap the compilation"
  ON
  "UNIX"
  OFF)

if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)

  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}"
        CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}"
        CACHE STRING "CXX compiler launcher")
  else()
    message(STATUS "Could not find ccache. \
                   Consider installing ccache to speed up compilation.")
  endif()
endif()

# ---[ Version numbers for generated libraries
file(READ version.txt TORCH_MUSA_DEFAULT_VERSION)

# Strip trailing newline
string(REGEX REPLACE "\n$" "" TORCH_MUSA_DEFAULT_VERSION
                     "${TORCH_MUSA_DEFAULT_VERSION}")

if("${TORCH_MUSA_DEFAULT_VERSION} " STREQUAL " ")
  message(WARNING "Could not get version from base 'version.txt'")

  # If we can't get the version from the version file we should probably set it
  # to something non-sensical like 0.0.0
  set(TORCH_MUSA_DEFAULT_VERSION, "0.0.0")
endif()

if(NOT GENERATED_PORTING_DIR)
  message(
    FATAL_ERROR
      "Generated porting musa files are needed for building torch_musa")
endif()

if(NOT CODE_GENERATED_DIR)
  message(FATAL_ERROR "MUSA codegen files are needed for building torch_musa")
endif()

include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)
torch_musa_parse_version_str(TORCH_MUSA ${TORCH_MUSA_DEFAULT_VERSION})
set(TORCH_MUSA_SOVERSION
    "${TORCH_MUSA_VERSION_MAJOR}.${TORCH_MUSA_VERSION_MINOR}")

set(TORCH_MUSA_PYTHON_MINIMUM_VERSION 3.9)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: \
         Debug Release RelWithDebInfo MinSizeRel Coverage." FORCE)
endif()

string(APPEND CMAKE_CXX_FLAGS " -O2 -fPIC")
string(APPEND CMAKE_CXX_FLAGS " -Wall")
string(APPEND CMAKE_CXX_FLAGS " -Wextra")
string(APPEND CMAKE_CXX_FLAGS " -Werror")
string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-parameter")
string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-variable")
string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-function")
string(APPEND CMAKE_CXX_FLAGS " -Wno-unknown-pragmas")
string(APPEND CMAKE_CXX_FLAGS " -Wno-attributes")
string(APPEND CMAKE_CXX_FLAGS " -Wno-redundant-move")
string(APPEND CMAKE_CXX_FLAGS " -Wno-sign-compare")
string(APPEND CMAKE_CXX_FLAGS " -Wno-missing-field-initializers")

# Use ld.gold if available, fall back to ld.bfd (the default ld) if not
if(USE_GOLD_LINKER)
  if(USE_DISTRIBUTED AND USE_MPI)
    message(WARNING "Refusing to use gold when USE_MPI=1")
  else()
    execute_process(COMMAND "${CMAKE_C_COMPILER}" -fuse-ld=gold -Wl,--version
                    ERROR_QUIET OUTPUT_VARIABLE LD_VERSION)

    if(NOT "${LD_VERSION}" MATCHES "GNU gold")
      message(
        WARNING
          "USE_GOLD_LINKER was set but ld.gold isn't available, turning it off")
      set(USE_GOLD_LINKER OFF)
    else()
      message(STATUS "ld.gold is available, using it to link")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
      set(CMAKE_SHARED_LINKER_FLAGS
          "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold")
      set(CMAKE_MODULE_LINKER_FLAGS
          "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=gold")
    endif()
  endif()
endif()
# These flags are not available in GCC-4.8.5. Set only when using clang.
# Compared against
# https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/Option-Summary.html
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  if(${COLORIZE_OUTPUT})
    string(APPEND CMAKE_CXX_FLAGS " -fcolor-diagnostics")
  endif()
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION
                                                 VERSION_GREATER 4.9)
  if(${COLORIZE_OUTPUT})
    string(APPEND CMAKE_CXX_FLAGS " -fdiagnostics-color=always")
  endif()
endif()

string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fno-omit-frame-pointer -O0")
string(APPEND CMAKE_LINKER_FLAGS_DEBUG " -fno-omit-frame-pointer -O0")
string(APPEND CMAKE_CXX_FLAGS " -fno-math-errno")
string(APPEND CMAKE_CXX_FLAGS " -fno-trapping-math")

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-Werror=format" HAS_WERROR_FORMAT)
if(HAS_WERROR_FORMAT)
  string(APPEND CMAKE_CXX_FLAGS " -Werror=format")
endif()

check_cxx_compiler_flag("-Wno-error=maybe-uninitialized"
                        HAS_WNO_ERROR_MAYBE_UNINITIALIZED)
if(HAS_WNO_ERROR_MAYBE_UNINITIALIZED)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-error=maybe-uninitialized")
endif()

check_cxx_compiler_flag("-Werror=cast-function-type"
                        HAS_WERROR_CAST_FUNCTION_TYPE)
if(HAS_WERROR_CAST_FUNCTION_TYPE)
  string(APPEND CMAKE_CXX_FLAGS " -Werror=cast-function-type")
endif()

check_cxx_compiler_flag("-Werror=sign-compare" HAS_WERROR_SIGN_COMPARE)

if(USE_ASAN)
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fsanitize=address")
  string(APPEND CMAKE_LINKER_FLAGS_DEBUG " -fsanitize=address")
endif()

if(USE_TSAN)
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fsanitize=thread")
  string(APPEND CMAKE_LINKER_FLAGS_DEBUG " -fsanitize=thread")
endif()

# =============================================================
set(PLUGIN_NAME "musa_python")
set(MUSA_CSRCS)
set(MUSA_KERNELS_LIB "musa_kernels")

set(DEPENDENT_LIBRARIES "")
set(DEPENDENT_INCLUDE_DIRS "")

# configure MUSA related things
if(NOT DEFINED ENV{MUSA_HOME})
  set(MUSA_PATH "/usr/local/musa")
  set(ENV{MUSA_HOME} "${MUSA_PATH}")
else()
  set(MUSA_PATH $ENV{MUSA_HOME})
endif()
list(APPEND CMAKE_MODULE_PATH "${MUSA_PATH}/cmake")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

# parse the version of MUSA, the version number is incorrect before MUSA 4.3.0
parse_real_musa_version(REAL_MUSA_VERSION)

find_package(MUSA REQUIRED)

if(MUSA_FOUND)
  # torch_musa/csrc/extension will also use TORCH_MUSA_ARCH, so parse mcc
  # compile arch flags here once.
  torch_musa_get_mcc_arch_list(TORCH_MUSA_ARCH_LIST)

  set(MAX_TORCH_MUSA_ARCH_VERSION 0)

  foreach(TORCH_MUSA_ARCH ${TORCH_MUSA_ARCH_LIST})
    string(APPEND MUSA_MCC_FLAGS " --offload-arch=mp_${TORCH_MUSA_ARCH}")

    string(REGEX MATCH "[0-9]+" ARCH_INT ${TORCH_MUSA_ARCH})
    math(EXPR ARCH_VERSION_INT "${ARCH_INT} * 10")
    if(ARCH_VERSION_INT GREATER MAX_TORCH_MUSA_ARCH_VERSION)
      set(MAX_TORCH_MUSA_ARCH_VERSION ${ARCH_VERSION_INT})
    endif()
  endforeach()

  # MUSA support mublasLt since 4.3.0
  if(REAL_MUSA_VERSION GREATER_EQUAL "4030")
    find_package(MUSAToolkit REQUIRED COMPONENTS musolver mublas mublasLt mufft)
    list(
      APPEND
      DEPENDENT_LIBRARIES
      MUSA::mufft
      MUSA::mublas
      MUSA::mublasLt
      MUSA::musolver)
  else()
    find_package(MUSAToolkit REQUIRED COMPONENTS musolver mublas mufft)
    list(APPEND DEPENDENT_LIBRARIES MUSA::mublas MUSA::musolver MUSA::mufft)
  endif()

  find_package(MUDNN REQUIRED)

  list(APPEND DEPENDENT_INCLUDE_DIRS ${MUSAToolkit_INCLUDE_DIR}
       ${MUDNN_INCLUDE_DIRS})
  list(APPEND DEPENDENT_LIBRARIES ${MUSA_LIBRARIES} ${MUDNN_LIBRARIES})

  if(USE_MCCL)
    if(NOT DEFINED ENV{MCCL_INCLUDE_DIR})
      set(ENV{MCCL_INCLUDE_DIR} "${MUSA_PATH}/include")
    endif()
    if(NOT DEFINED ENV{MCCL_LIB_DIR})
      set(ENV{MCCL_LIB_DIR} "${MUSA_PATH}/lib")
    endif()
    find_package(MCCL REQUIRED)

    list(APPEND DEPENDENT_INCLUDE_DIRS ${MCCL_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${MCCL_LIBRARIES})
    add_definitions(-DUSE_MCCL)

    # For mccl BF16 compilation
    add_definitions(-DMARCH_TYPE=${MAX_TORCH_MUSA_ARCH_VERSION})
  endif()

  set(MUSA_LINK_LIBRARIES_KEYWORD PUBLIC)
else()
  message(FATAL_ERROR "MUSA not found")
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/torch_musa/csrc/aten/core)
include_directories(${PROJECT_SOURCE_DIR}/torch_musa/csrc/aten/utils)
include_directories(${PROJECT_SOURCE_DIR}/torch_musa/csrc/aten/ops)
include_directories(${PROJECT_SOURCE_DIR}/torch_musa/csrc/aten/ops/musa)
include_directories(${DEPENDENT_INCLUDE_DIRS})

# Set PyTorch headers dir
include_directories(${GENERATED_PORTING_DIR}/include)
include_directories(${GENERATED_PORTING_DIR}/aten/src)
include_directories(${GENERATED_PORTING_DIR}/include/torch/csrc/api/include)
include_directories(${CODE_GENERATED_DIR})

# Set Python include dir, python version shoud be ge than 3.8
find_package(Python3 ${TORCH_MUSA_PYTHON_MINIMUM_VERSION} REQUIRED
             COMPONENTS Development Interpreter)

if(Python3_FOUND)
  include_directories(${Python3_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Cannot find installed Python3 header file directory")
endif()

set(CODEGEN_TARGET "musa_codegen")
set(GENERATED_MUSA_CSRCS
    "${CODE_GENERATED_DIR}/ATen/RegisterMUSA.cpp"
    "${CODE_GENERATED_DIR}/ATen/RegisterQuantizedMUSA.cpp"
    "${CODE_GENERATED_DIR}/ATen/RegisterSparseMUSA.cpp"
    "${CODE_GENERATED_DIR}/ATen/RegisterSparseCsrMUSA.cpp"
    "${PROJECT_SOURCE_DIR}/torch_musa/csrc/inductor/aoti_torch/generated/c_shim_musa.cpp"
)
add_custom_command(
  OUTPUT ${GENERATED_MUSA_CSRCS}
  COMMENT "Generating for musa op files ..."
  COMMAND python tools/codegen/gen_torch_musa.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${PROJECT_SOURCE_DIR}/torch_musa/csrc/aten/ops/musa_functions.yaml
          ${BUILD_PYTORCH_REPO_PATH}/aten/src/ATen/native/native_functions.yaml
          ${BUILD_PYTORCH_REPO_PATH}/tools/autograd/derivatives.yaml
  VERBATIM)
set_source_files_properties(${GENERATED_MUSA_CSRCS} PROPERTIES GENERATED TRUE)
add_custom_target(${CODEGEN_TARGET} DEPENDS ${GENERATED_MUSA_CSRCS})

# add AOT Inductor fallback op copy
set(FALLBACK_OPS_PY "${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/fallback_ops.py")
set(FALLBACK_OPS_OUT
    "${PROJECT_SOURCE_DIR}/torch_musa/_inductor/codegen/fallback_ops.py")

add_custom_command(
  OUTPUT ${FALLBACK_OPS_OUT}
  COMMENT "Copying fallback_ops.py to ${FALLBACK_OPS_OUT} ..."
  COMMAND ${CMAKE_COMMAND} -E copy ${FALLBACK_OPS_PY} ${FALLBACK_OPS_OUT}
  DEPENDS ${FALLBACK_OPS_PY}
  VERBATIM)

set_source_files_properties(${FALLBACK_OPS_OUT} PROPERTIES GENERATED TRUE)

add_custom_target(fallback_codegen DEPENDS ${FALLBACK_OPS_OUT})

add_dependencies(${CODEGEN_TARGET} fallback_codegen)
# add AOT Inductor end.

add_subdirectory(${PROJECT_SOURCE_DIR}/torch_musa/csrc)
set(CPP_SRCS ${MUSA_CSRCS})
add_library(${PLUGIN_NAME} SHARED ${CPP_SRCS})

# keep consistent with PyTorch's ATen generating library, according to
# Codegen.cmake
if(USE_PER_OPERATOR_HEADERS)
  target_compile_definitions(${MUSA_KERNELS_LIB} PUBLIC AT_PER_OPERATOR_HEADERS)
  target_compile_definitions(${PLUGIN_NAME} PUBLIC AT_PER_OPERATOR_HEADERS)
endif()

target_compile_definitions(${MUSA_KERNELS_LIB}
                           PUBLIC -DREAL_MUSA_VERSION=${REAL_MUSA_VERSION})
target_compile_definitions(${PLUGIN_NAME}
                           PUBLIC -DREAL_MUSA_VERSION=${REAL_MUSA_VERSION})

add_dependencies(${MUSA_KERNELS_LIB} ${CODEGEN_TARGET})

string(REPLACE ";" " " PYTORCH_MUSA_ARCH_readable "${TORCH_MUSA_ARCH_LIST}")
set_source_files_properties(
  ${PROJECT_SOURCE_DIR}/torch_musa/csrc/core/Module.cpp
  PROPERTIES COMPILE_FLAGS
             "-DMUSA_ARCH_FLAGS=\"${PYTORCH_MUSA_ARCH_readable}\"")

target_link_libraries(${PLUGIN_NAME} PUBLIC ${DEPENDENT_LIBRARIES})
target_link_libraries(${PLUGIN_NAME} PUBLIC ${MUSA_KERNELS_LIB})

if(LINUX)
  target_link_libraries(${PLUGIN_NAME} PRIVATE dl)
endif()

# compile musa_kernel of the extension
set(EXT_MUSA_KERNEL_LIB "_ext_musa_kernels")
add_subdirectory(${PROJECT_SOURCE_DIR}/torch_musa/csrc/extension)
target_link_libraries(${EXT_MUSA_KERNEL_LIB} PUBLIC ${PLUGIN_NAME})

set(INSTALL_BIN_DIR "bin")
set(INSTALL_LIB_DIR "lib64")
set(INSTALL_INC_DIR "include")
set(INSTALL_SHARE_DIR "share")
set(INSTALL_DOC_DIR "docs")

set_target_properties(
  ${PLUGIN_NAME} PROPERTIES VERSION ${TORCH_MUSA_VERSION}
                            SOVERSION ${TORCH_MUSA_VERSION_MAJOR})
set_target_properties(
  ${PLUGIN_NAME}
  PROPERTIES OUTPUT_NAME ${PLUGIN_NAME}
             POSITION_INDEPENDENT_CODE true
             INSTALL_RPATH_USE_LINK_PATH false
             RUNTIME_OUTPUT_DIRECTORY ${INSTALL_BIN_DIR}
             LIBRARY_OUTPUT_DIRECTORY ${INSTALL_LIB_DIR}
             ARCHIVE_OUTPUT_DIRECTORY ${INSTALL_LIB_DIR})
set_target_properties(${PLUGIN_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
set_target_properties(${PLUGIN_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN")

install(TARGETS ${PLUGIN_NAME})

include(${PROJECT_SOURCE_DIR}/cmake/summary.cmake)
torch_musa_build_configuration_summary()

configure_file(${PROJECT_SOURCE_DIR}/cmake/TorchMusaConfig.cmake.in
               ${PROJECT_BINARY_DIR}/TorchMusaConfig.cmake COPYONLY)
install(FILES ${PROJECT_BINARY_DIR}/TorchMusaConfig.cmake
        DESTINATION share/cmake/TorchMusa)

install(FILES ${PROJECT_SOURCE_DIR}/cmake/modules/FindMUDNN.cmake
              ${PROJECT_SOURCE_DIR}/cmake/modules/FindMUSAToolkits.cmake
        DESTINATION share/cmake/modules/)

install(FILES ${PROJECT_SOURCE_DIR}/cmake/cpu_flags.cmake
              ${PROJECT_SOURCE_DIR}/cmake/utils.cmake
              ${PROJECT_SOURCE_DIR}/cmake/summary.cmake DESTINATION share/cmake)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/build/generated_cuda_compatible
        DESTINATION share/)

install(DIRECTORY ${CODE_GENERATED_DIR} DESTINATION share/)
