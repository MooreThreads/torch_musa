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
