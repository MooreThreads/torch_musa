# Find the musa_runtime libraries
#
# The following variables are optionally searched for defaults
#  MUSA_RUNTIME_INCLUDE_DIR: Directory where MUSA_RUNTIME header is found
#  MUSA_RUNTIME_LIB_DIR: Directory where MUSA_RUNTIME library is found
#
# The following are set after configuration is done:
#  MUSA_RUNTIME_FOUND
#  MUSA_RUNTIME_INCLUDE_DIRS
#  MUSA_RUNTIME_LIBRARIES

include(FindPackageHandleStandardArgs)

SET(MUSAToolkits_INCLUDE_SEARCH_PATHS $ENV{MUSATOOLKITS_PATH}/include)
SET(MUSAToolkits_LIB_SEARCH_PATHS $ENV{MUSATOOLKITS_PATH}/lib)

find_path(MUSAToolkits_INCLUDE_DIR NAMES musa_runtime.h
          PATHS ${MUSAToolkits_INCLUDE_SEARCH_PATHS}
          NO_DEFAULT_PATH)
find_path(MUSAToolkits_INCLUDE_DIR NAMES musa_runtime.h
          NO_CMAKE_FIND_ROOT_PATH)

find_library(MUSAToolkits_LIBRARY NAMES musart
          PATHS ${MUSAToolkits_LIB_SEARCH_PATHS}
          NO_DEFAULT_PATH)
find_library(MUSAToolkits_LIBRARY NAMES musart
          NO_CMAKE_FIND_ROOT_PATH)

find_package_handle_standard_args(MUSAToolkits DEFAULT_MSG MUSAToolkits_INCLUDE_DIR MUSAToolkits_LIBRARY)

if(MUSAToolkits_FOUND)
  set(MUSAToolkits_INCLUDE_DIRS ${MUSAToolkits_INCLUDE_DIR})
  set(MUSAToolkits_LIBRARIES ${MUSAToolkits_LIBRARY})

  mark_as_advanced(MUSAToolkits_ROOT_DIR MUSAToolkits_LIBRARY_RELEASE MUSAToolkits_LIBRARY_DEBUG
      MUSAToolkits_LIBRARY MUSAToolkits_INCLUDE_DIR )
endif()
