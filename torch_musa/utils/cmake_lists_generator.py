"""Generator for CMakeLists.txt"""

from os.path import realpath, dirname, join
from typing import List, Tuple


class CMakeListsGenerator:
    """
    Generator for CMakeLists.txt for musa backend project
    """

    def __init__(
        self,
        sources: List[str],
        include_dirs: List[str],
        link_libraries: List[str],
        define_macros: List[Tuple] = None,
        project_name: str = "DEMO_MUSA",
        plugin_name: str = "demo_musa",
        customized_cmake_lists: str = None,
    ):
        self.project_name = project_name
        self.plugin_name = plugin_name
        self.define_macros = define_macros
        self.sources = sources
        self.include_dirs = include_dirs
        self.link_libraries = link_libraries
        self.cmake_lists = customized_cmake_lists

    def generate(self):
        self.fill_in_cmake_lists_template()
        with open("CMakeLists.txt", "w", encoding="utf-8") as f:
            f.write(self.cmake_lists)

    def fill_in_cmake_lists_template(self):
        """Filling in customized values for cmake lists template"""
        if self.cmake_lists is None:
            torch_musa_path = dirname(dirname(__file__))
            sources_paths = "\n".join([realpath(source) for source in self.sources])
            if not sources_paths:
                raise RuntimeError("Please provide sources.")
            include_dir_paths = "\n".join(
                [realpath(include_dir) for include_dir in self.include_dirs]
            )
            link_libraries_paths = "\n".join(
                [realpath(library) for library in self.link_libraries]
            )

            temp_define_macros = []
            temp_define_mcc_macros = []
            for k, v in self.define_macros:
                if v is None:
                    temp_define_macros += [f"-D{k}"]
                    temp_define_mcc_macros += [f'" -D{k}"']
                else:
                    temp_define_macros += [f"-D{k}={v}"]
                    temp_define_mcc_macros += [f'" -D{k}={v}"']
            define_macros = "\n".join(temp_define_macros)
            define_mcc_macros = "\n".join(temp_define_mcc_macros)

            self.cmake_lists = "".join(
                [
                    """cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
                """,
                    f"""
project({self.project_name} CXX C)
                """,
                    """
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LINUX TRUE)
else()
    message(FATAL_ERROR, "could be built only on Linux now!")
endif()
                """,
                    """
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)

if(env_cxx_standard GREATER -1) 
    message( 
        WARNING "C++ standard version definition detected in environment variable." 
        "PyTorch requires -std=c++17. Please remove -std=c++ settings in your environment.")
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING 
"The C++ standard whose features are requested to build this target.")

set(CMAKE_C_STANDARD   11 CACHE STRING 
"The C standard whose features are requested to build this target.")

if(DEFINED GLIBCXX_USE_CXX11_ABI)
    if(${GLIBCXX_USE_CXX11_ABI} EQUAL 1)
        set(CXX_STANDARD_REQUIRED ON)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

option(ENABLE_COMPILE_FP64 "Enable FP64" OFF)
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
                """,
                    f"""
include({torch_musa_path}/share/cmake/utils.cmake)
                """,
                    """
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set - defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING 
    "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage." FORCE)
endif() 

string(APPEND CMAKE_CXX_FLAGS " -O2 -fPIC")
string(APPEND CMAKE_CXX_FLAGS " -Wall")
string(APPEND CMAKE_CXX_FLAGS " -Wextra")
string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-parameter")
string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-variable")
string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-function")
string(APPEND CMAKE_CXX_FLAGS " -Wno-sign-compare")
string(APPEND CMAKE_CXX_FLAGS " -Wno-missing-field-initializers")

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
                """,
                    """
if(DEFINED ENV{MUSA_ARCH})
  set(MUSA_ARCH $ENV{MUSA_ARCH})
  string(APPEND MUSA_MCC_FLAGS " --cuda-gpu-arch=mp_${MUSA_ARCH}")

  if(${MUSA_ARCH} EQUAL 21)
    add_definitions(-DTORCH_MUSA_ARCH=210)
  elseif(${MUSA_ARCH} EQUAL 22)
    add_definitions(-DTORCH_MUSA_ARCH=220)
  elseif(${MUSA_ARCH} EQUAL 31)
    add_definitions(-DTORCH_MUSA_ARCH=310)
  else()
    add_definitions(-DTORCH_MUSA_ARCH=110)
  endif()
endif()
""",
                    (
                        f"""
add_definitions(
{define_macros}
)
"""
                        if define_macros
                        else ""
                    ),
                    f"""
set(PLUGIN_NAME "{self.plugin_name}")
                """,
                    f"""
set(MUSA_CSRCS)
set(CMAKE_MODULE_PATH {torch_musa_path}/share/cmake/modules)
set(DEPENDENT_LIBRARIES "")
set(DEPENDENT_INCLUDE_DIRS "")
find_package(MUDNN)
                """,
                    """
if(MUDNN_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${MUDNN_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${MUDNN_LIBRARIES})
else()
    message(WARNING " The environment variable MUSA_HOME may be not specified." 
    "Using default MUDNN PATH: /usr/local/musa")
    
    list(APPEND DEPENDENT_INCLUDE_DIRS "/usr/local/musa/include")
    list(APPEND DEPENDENT_LIBRARIES "/usr/local/musa/lib/libmudnn.so")
    set(MUDNN_PATH "/usr/local/musa")
    set(MUDNN_LIBRARIES "/usr/local/musa/lib/libmudnn.so")
endif()

find_package(MUSAToolkits)

if(MUSAToolkits_FOUND)
    list(APPEND DEPENDENT_INCLUDE_DIRS ${MUSAToolkits_INCLUDE_DIRS})
    list(APPEND DEPENDENT_LIBRARIES ${MUSAToolkits_LIBRARIES})
else()
    message(WARNING " The environment variable MUSA_HOME may be not specified." 
    "Using default MUSATOOLKITS PATH: /usr/local/musa")
    
    list(APPEND DEPENDENT_INCLUDE_DIRS "/usr/local/musa/include/")
    list(APPEND DEPENDENT_LIBRARIES "/usr/local/musa/lib/libmusart.so")
    set(ENV{MUSA_HOME} "/usr/local/musa")
    set(MUSATOOLKITS_PATH "/usr/local/musa")
    set(MUSAToolkits_LIBRARIES "/usr/local/musa/lib/")
endif()

if(DEFINED PYTHON_INCLUDE_DIR)
    include_directories(${PYTHON_INCLUDE_DIR})
else()
    message(FATAL_ERROR, "Cannot find installed Python head file directory")
endif()

list(APPEND CMAKE_MODULE_PATH $ENV{MUSA_HOME}/cmake)
find_package(MUSA REQUIRED)
                """,
                    f"""
FILE(GLOB MU_SRCS
{sources_paths}
)
                """,
                    """
append_cxx_flag_if_supported("-Wno-unused-parameter" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-unused-variable" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-Wno-sign-compare" CMAKE_CXX_FLAGS)
append_cxx_flag_if_supported("-w" CMAKE_CXX_FLAGS)
""",
                    f"""
string(APPEND MUSA_MCC_FLAGS 
{define_mcc_macros}
)
string(APPEND MUSA_MCC_FLAGS " -U__CUDA__")

set(MUSA_VERBOSE_BUILD ON)

                """,
                    (
                        f"""
musa_include_directories(
{include_dir_paths}
)
                """
                        if include_dir_paths
                        else ""
                    ),
                    "\n",
                    f"musa_add_library({self.plugin_name}" + " STATIC ${MU_SRCS})",
                    """
set(INSTALL_BIN_DIR "bin")
set(INSTALL_LIB_DIR "lib64")
set(INSTALL_INC_DIR "include")
set(INSTALL_SHARE_DIR "share")
set(INSTALL_DOC_DIR "docs")
                """,
                    f"""
set_target_properties({self.plugin_name} PROPERTIES
OUTPUT_NAME {self.plugin_name}
                """,
                    """
POSITION_INDEPENDENT_CODE true
INSTALL_RPATH_USE_LINK_PATH false
RUNTIME_OUTPUT_DIRECTORY ${INSTALL_BIN_DIR}
LIBRARY_OUTPUT_DIRECTORY ${INSTALL_LIB_DIR}
ARCHIVE_OUTPUT_DIRECTORY ${INSTALL_LIB_DIR}
)
                
set(CUBLAS_LIB $ENV{MUSA_HOME}/lib/libmublas.so)
                """,
                    "\n",
                    (
                        f"""
target_link_libraries({self.plugin_name} 
{link_libraries_paths}
)
                """
                        if link_libraries_paths
                        else ""
                    ),
                    "\n",
                    f"target_link_libraries({self.plugin_name}"
                    + " ${DEPENDENT_LIBRARIES})",
                    "\n",
                    f"target_link_libraries({self.plugin_name}" + " ${CUBLAS_LIB})",
                    "\n",
                    f"""
target_link_libraries({self.plugin_name} "{join(torch_musa_path, 
"lib/libmusa_python.so")}")

install(TARGETS {self.plugin_name})
                """,
                    f"""
include({torch_musa_path}/share/cmake/summary.cmake)
torch_musa_build_configuration_summary()
""",
                ]
            )
