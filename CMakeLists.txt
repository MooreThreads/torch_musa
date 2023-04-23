cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(Torch_MUSA CXX C)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(LINUX TRUE)
else()
  message(FATAL_ERROR, "torch_musa could be built only on Linux now!")
endif()

set(CMAKE_INSTALL_MESSAGE NEVER)

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
  message(
      WARNING "C++ standard version definition detected in environment variable."
      "PyTorch requires -std=c++17. Please remove -std=c++ settings in your environment.")
endif()
set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard whose features are requested to build this target.")
set(CMAKE_C_STANDARD   11 CACHE STRING "The C standard whose features are requested to build this target.")

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

option(USE_ASAN "Use Address Sanitizer" OFF)
option(USE_TSAN "Use Thread Sanitizer" OFF)
option(USE_MUSA "Use MUSA" ON)
include(CMakeDependentOption)
cmake_dependent_option(
	USE_MUDNN "Use muDNN" ON
	"USE_MUSA" OFF)

cmake_dependent_option(USE_CCACHE "Attempt using CCache to wrap the compilation" ON "UNIX" OFF)
if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
  else()
    message(STATUS "Could not find ccache. Consider installing ccache to speed up compilation.")
  endif()
endif()

# ---[ Version numbers for generated libraries
file(READ version.txt TORCH_MUSA_DEFAULT_VERSION)
# Strip trailing newline
string(REGEX REPLACE "\n$" "" TORCH_MUSA_DEFAULT_VERSION "${TORCH_MUSA_DEFAULT_VERSION}")
if("${TORCH_MUSA_DEFAULT_VERSION} " STREQUAL " ")
  message(WARNING "Could not get version from base 'version.txt'")
  # If we can't get the version from the version file we should probably
  # set it to something non-sensical like 0.0.0
  set(TORCH_MUSA_DEFAULT_VERSION, "0.0.0")
endif()
include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)
torch_musa_parse_version_str(TORCH_MUSA ${TORCH_MUSA_DEFAULT_VERSION})
set(TORCH_MUSA_SOVERSION "${TORCH_MUSA_VERSION_MAJOR}.${TORCH_MUSA_VERSION_MINOR}")

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set - defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage." FORCE)
endif()

string(APPEND CMAKE_CXX_FLAGS " -O2 -fPIC")
string(APPEND CMAKE_CXX_FLAGS " -Wall")
string(APPEND CMAKE_CXX_FLAGS " -Wextra")
string(APPEND CMAKE_CXX_FLAGS " -Werror")
string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-parameter")
string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-variable")
string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-function")
string(APPEND CMAKE_CXX_FLAGS " -Wno-sign-compare")
string(APPEND CMAKE_CXX_FLAGS " -Wno-missing-field-initializers")

# Use ld.gold if available, fall back to ld.bfd (the default ld) if not
if(USE_GOLD_LINKER)
  if(USE_DISTRIBUTED AND USE_MPI)
    message(WARNING "Refusing to use gold when USE_MPI=1")
  else()
    execute_process(
      COMMAND
      "${CMAKE_C_COMPILER}" -fuse-ld=gold -Wl,--version
       ERROR_QUIET
       OUTPUT_VARIABLE LD_VERSION)
    if(NOT "${LD_VERSION}" MATCHES "GNU gold")
      message(WARNING "USE_GOLD_LINKER was set but ld.gold isn't available, turning it off")
      set(USE_GOLD_LINKER OFF)
    else()
      message(STATUS "ld.gold is available, using it to link")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold")
      set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=gold")
    endif()
  endif()
endif()

# These flags are not available in GCC-4.8.5. Set only when using clang.
# Compared against https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/Option-Summary.html
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  if(${COLORIZE_OUTPUT})
    string(APPEND CMAKE_CXX_FLAGS " -fcolor-diagnostics")
  endif()
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
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
check_cxx_compiler_flag("-Werror=cast-function-type" HAS_WERROR_CAST_FUNCTION_TYPE)
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

#=============================================================

set(PLUGIN_NAME "musa_python")
set(MUSA_CSRCS)
set(MUSA_KERNELS_LIB "musa_kernels")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
set(DEPENDENT_LIBRARIES "")
set(DEPENDENT_INCLUDE_DIRS "")
find_package(MUDNN)
if(MUDNN_FOUND)
  list(APPEND DEPENDENT_INCLUDE_DIRS ${MUDNN_INCLUDE_DIRS})
  list(APPEND DEPENDENT_LIBRARIES ${MUDNN_LIBRARIES})
else()
  # set default mudnn library path
  message(WARNING, "Cannot find MUDNN, using default MUDNN_PATH=/usr/local/mudnn.")
  list(APPEND DEPENDENT_INCLUDE_DIRS "/usr/local/mudnn/include/cc")
  list(APPEND DEPENDENT_LIBRARIES "/usr/local/mudnn/lib64/libmudnn.so")
  set(ENV{MUDNN_PATH} "/usr/local/mudnn")
  set(MUDNN_LIBRARIES "/usr/local/musa/lib64/libmudnn.so")
endif()

find_package(MUSAToolkits)
if(MUSAToolkits_FOUND)
  list(APPEND DEPENDENT_INCLUDE_DIRS ${MUSAToolkits_INCLUDE_DIRS})
  list(APPEND DEPENDENT_LIBRARIES ${MUSAToolkits_LIBRARIES})
else()
  # set default musa_toolkits path
  message(WARNING, "Cannot find MUSAToolkits, using default MUSATOOLKITS_PATH=/usr/local/musa")
  list(APPEND DEPENDENT_INCLUDE_DIRS "/usr/local/musa/include/")
  list(APPEND DEPENDENT_LIBRARIES "/usr/local/musa/lib/libmusart.so")
  set(ENV{MUSATOOLKITS_PATH} "/usr/local/musa")
  set(MUSAToolkits_LIBRARIES "/usr/local/musa/lib/")
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/torch_musa/csrc/aten/core)
include_directories(${PROJECT_SOURCE_DIR}/torch_musa/csrc/aten/utils)
include_directories(${PROJECT_SOURCE_DIR}/torch_musa/csrc/aten/ops)
include_directories(${PROJECT_SOURCE_DIR}/torch_musa/csrc/aten/ops/musa)
include_directories(${DEPENDENT_INCLUDE_DIRS})

# Set installed PyTorch dir
if(DEFINED PYTORCH_INSTALL_DIR)
  include_directories(${PYTORCH_INSTALL_DIR}/include)
  include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include)
else()
  message(FATAL_ERROR, "Cannot find installed PyTorch directory")
endif()

# Set Python include dir
if(DEFINED PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR, "Cannot find installed Python head file directory")
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/torch_musa/csrc)
set(CPP_SRCS ${MUSA_CSRCS})
add_library(${PLUGIN_NAME} SHARED ${CPP_SRCS})

target_link_libraries(${PLUGIN_NAME} PUBLIC ${DEPENDENT_LIBRARIES})
target_link_libraries(${PLUGIN_NAME} PUBLIC ${MUSA_KERNELS_LIB})

set(INSTALL_BIN_DIR "bin")
set(INSTALL_LIB_DIR "lib64")
set(INSTALL_INC_DIR "include")
set(INSTALL_SHARE_DIR "share")
set(INSTALL_DOC_DIR "docs")

set_target_properties(${PLUGIN_NAME} PROPERTIES
  VERSION ${TORCH_MUSA_DEFAULT_VERSION}
  SOVERSION ${TORCH_MUSA_VERSION_MAJOR}
  )
set_target_properties(${PLUGIN_NAME} PROPERTIES
  OUTPUT_NAME ${PLUGIN_NAME}
  POSITION_INDEPENDENT_CODE true
  INSTALL_RPATH_USE_LINK_PATH false
  RUNTIME_OUTPUT_DIRECTORY ${INSTALL_BIN_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${INSTALL_LIB_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${INSTALL_LIB_DIR}
  )

install(TARGETS ${PLUGIN_NAME})

include(${PROJECT_SOURCE_DIR}/cmake/summary.cmake)
torch_musa_build_configuration_summary()
