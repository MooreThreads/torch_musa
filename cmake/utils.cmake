# cmake-format: off
# Parses a version string that might have values beyond major, minor, and patch
# and set version variables for the library. 
# Usage:
# torch_musa_parse_version_str(<library_name> <version_string>)
# cmake-format: on
function(torch_musa_parse_version_str lib_name version_str)
  string(REGEX REPLACE "^([0-9]+).*$" "\\1" ${lib_name}_VERSION_MAJOR
                       "${version_str}")
  string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*$" "\\1" ${lib_name}_VERSION_MINOR
                       "${version_str}")
  string(REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1"
                       ${lib_name}_VERSION_PATCH "${version_str}")
  set(${lib_name}_VERSION_MAJOR ${${lib_name}_VERSION_MAJOR} ${ARGN}
      PARENT_SCOPE)
  set(${lib_name}_VERSION_MINOR ${${lib_name}_VERSION_MINOR} ${ARGN}
      PARENT_SCOPE)
  set(${lib_name}_VERSION_PATCH ${${lib_name}_VERSION_PATCH} ${ARGN}
      PARENT_SCOPE)
  set(${lib_name}_VERSION
      "${${lib_name}_VERSION_MAJOR}.${${lib_name}_VERSION_MINOR}.${${lib_name}_VERSION_PATCH}"
      PARENT_SCOPE)
endfunction()

function(append_cxx_flag_if_supported flag outputvar)
  string(TOUPPER "HAS${flag}" _FLAG_NAME)
  string(REGEX REPLACE "[=-]" "_" _FLAG_NAME "${_FLAG_NAME}")
  check_cxx_compiler_flag("${flag}" ${_FLAG_NAME})
  if(${_FLAG_NAME})
    string(APPEND ${outputvar} " ${flag}")
    set(${outputvar} "${${outputvar}}" PARENT_SCOPE)
  endif()
endfunction()
