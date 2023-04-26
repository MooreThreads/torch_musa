# Parses a version string that might have values beyond major, minor, and patch
# and set version variables for the library.
# Usage:
#   torch_musa_parse_version_str(<library_name> <version_string>)
function(torch_musa_parse_version_str LIBNAME VERSIONSTR)
  string(REGEX REPLACE "^([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MAJOR "${VERSIONSTR}")
  string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MINOR  "${VERSIONSTR}")
  string(REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_PATCH "${VERSIONSTR}")
  set(${LIBNAME}_VERSION_MAJOR ${${LIBNAME}_VERSION_MAJOR} ${ARGN} PARENT_SCOPE)
  set(${LIBNAME}_VERSION_MINOR ${${LIBNAME}_VERSION_MINOR} ${ARGN} PARENT_SCOPE)
  set(${LIBNAME}_VERSION_PATCH ${${LIBNAME}_VERSION_PATCH} ${ARGN} PARENT_SCOPE)
  set(${LIBNAME}_VERSION "${${LIBNAME}_VERSION_MAJOR}.${${LIBNAME}_VERSION_MINOR}.${${LIBNAME}_VERSION_PATCH}" PARENT_SCOPE)
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
