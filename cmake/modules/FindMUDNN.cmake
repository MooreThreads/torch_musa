# Find the mudnn libraries
#
# The following variables are optionally searched for defaults
# cmake-format: off
# MUDNN_INCLUDE_DIR: 
# Directory where MUDNN header is found 
# MUDNN_LIB_DIR:
# Directory where MUDNN library is found
#
# The following are set after configuration is done:
# MUDNN_FOUND
# MUDNN_INCLUDE_DIRS 
# MUDNN_LIBRARIES
# cmake-format: on

include(FindPackageHandleStandardArgs)

set(MUDNN_INCLUDE_SEARCH_PATHS $ENV{MUSA_HOME}/include)
set(MUDNN_LIB_SEARCH_PATHS $ENV{MUSA_HOME}/lib)

macro(find_header var dir file)
  find_path(
    ${var}
    NAMES ${file}
    PATHS ${dir}
    NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
endmacro()

macro(find_lib var dir file)
  find_library(
    ${var}
    NAMES ${file}
    PATHS ${dir}
    NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
endmacro()

set(MUDNN_INCLUDE_DIR)
set(MUDNN_LIBRARY)
set(MUDNN_CXX_INCLUDE_DIR)
set(MUDNN_CXX_LIBRARY)

if(NOT USE_MUDNN_C_API)
  find_header(MUDNN_CXX_INCLUDE_DIR "${MUDNN_INCLUDE_SEARCH_PATHS}/mudnncxx"
              "mudnn.h")
  find_lib(MUDNN_CXX_LIBRARY ${MUDNN_LIB_SEARCH_PATHS} "mudnncxx")
  set(MUDNN_INCLUDE_DIR ${MUDNN_CXX_INCLUDE_DIR})
  set(MUDNN_LIBRARY ${MUDNN_CXX_LIBRARY})
endif()

if(NOT MUDNN_INCLUDE_DIR)
  find_header(MUDNN_INCLUDE_DIR ${MUDNN_INCLUDE_SEARCH_PATHS} "mudnn.h")
endif()

if(NOT MUDNN_LIBRARY)
  find_lib(MUDNN_LIBRARY ${MUDNN_LIB_SEARCH_PATHS} "mudnn")
endif()

find_package_handle_standard_args(MUDNN DEFAULT_MSG MUDNN_INCLUDE_DIR
                                  MUDNN_LIBRARY)

if(MUDNN_FOUND)
  set(MUDNN_INCLUDE_DIRS ${MUDNN_INCLUDE_DIR})
  set(MUDNN_LIBRARIES ${MUDNN_LIBRARY})
  set(MUDNN_PATH $ENV{MUSA_HOME})

  mark_as_advanced(
    MUDNN_ROOT_DIR
    MUDNN_CXX_INCLUDE_DIR
    MUDNN_CXX_LIBRARY
    MUDNN_LIBRARY_RELEASE
    MUDNN_LIBRARY_DEBUG
    MUDNN_LIBRARY
    MUDNN_INCLUDE_DIR)
endif()
