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

set(MUDNN_INCLUDE_SEARCH_PATHS $ENV{MUSA_HOME}/include/)
set(MUDNN_LIB_SEARCH_PATHS $ENV{MUSA_HOME}/lib)

find_path(
  MUDNN_INCLUDE_DIR
  NAMES mudnn.h
  PATHS ${MUDNN_INCLUDE_SEARCH_PATHS}
  NO_DEFAULT_PATH)
find_path(MUDNN_INCLUDE_DIR NAMES mudnn.h NO_CMAKE_FIND_ROOT_PATH)

find_library(
  MUDNN_LIBRARY
  NAMES mudnn
  PATHS ${MUDNN_LIB_SEARCH_PATHS}
  NO_DEFAULT_PATH)
find_library(MUDNN_LIBRARY NAMES mudnn NO_CMAKE_FIND_ROOT_PATH)

find_package_handle_standard_args(MUDNN DEFAULT_MSG MUDNN_INCLUDE_DIR
                                  MUDNN_LIBRARY)

if(MUDNN_FOUND)
  set(MUDNN_INCLUDE_DIRS ${MUDNN_INCLUDE_DIR})
  set(MUDNN_LIBRARIES ${MUDNN_LIBRARY})
  set(MUDNN_PATH $ENV{MUSA_HOME})

  mark_as_advanced(
    MUDNN_ROOT_DIR
    MUDNN_LIBRARY_RELEASE
    MUDNN_LIBRARY_DEBUG
    MUDNN_LIBRARY
    MUDNN_INCLUDE_DIR)
endif()
