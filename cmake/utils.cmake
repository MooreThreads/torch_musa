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

function(parse_real_musa_version outputvar)
  find_program(
    MUSA_TOOLKIT_VERSION_EXECUTABLE
    NAMES musa_toolkits_version
    PATHS "$ENV{MUSA_HOME}"
    PATH_SUFFIXES bin bin64
    NO_DEFAULT_PATH)
  mark_as_advanced(MUSA_TOOLKIT_VERSION_EXECUTABLE)
  if(MUSA_TOOLKIT_VERSION_EXECUTABLE)
    execute_process(COMMAND ${MUSA_TOOLKIT_VERSION_EXECUTABLE}
                    OUTPUT_VARIABLE MUSA_TOOLKITS_VERSION)
    string(REGEX REPLACE ".*\"([0-9]+)\\.([0-9]+)\\.([0-9]+)\".*" "\\1"
                         MUSA_TOOLKITS_VERSION_MAJOR ${MUSA_TOOLKITS_VERSION})
    string(REGEX REPLACE ".*\"([0-9]+)\\.([0-9]+)\\.([0-9]+)\".*" "\\2"
                         MUSA_TOOLKITS_VERSION_MINOR ${MUSA_TOOLKITS_VERSION})
    string(REGEX REPLACE ".*\"([0-9]+)\\.([0-9]+)\\.([0-9]+)\".*" "\\3"
                         MUSA_TOOLKITS_VERSION_PATCH ${MUSA_TOOLKITS_VERSION})

    math(
      EXPR
      MUSA_TOOLKITS_VERSION_INT
      "${MUSA_TOOLKITS_VERSION_MAJOR} * 1000 + ${MUSA_TOOLKITS_VERSION_MINOR} * 10 + ${MUSA_TOOLKITS_VERSION_PATCH}"
    )
    set(${outputvar} "${MUSA_TOOLKITS_VERSION_INT}" PARENT_SCOPE)
    mark_as_advanced(${outputvar})
  endif()
endfunction()

# Return the MUSA arch flags, if not specified by TORCH_MUSA_ARCH_LIST, then
# auto detect the arch of installed GPUs. Usage:
# torch_musa_get_mcc_arch_list(output_var)
macro(torch_musa_get_mcc_arch_list output_var)
  if(DEFINED ENV{TORCH_MUSA_ARCH_LIST})
    set(_TMP $ENV{TORCH_MUSA_ARCH_LIST})
  else()
    # Auto detect the arch of installed GPUs
    execute_process(
      COMMAND "musaInfo"
      COMMAND
        bash "-c"
        "grep -E 'major|minor'  | awk '{print $2}' | paste -d \"\" - - | sort -u | paste -sd \";\" | tr -d '\n'"
      RESULT_VARIABLE MUSA_INFO_RESULT
      OUTPUT_VARIABLE MUSA_ARCH_INSTALLED)
    if(NOT MUSA_INFO_RESULT EQUAL 0)
      message(
        FATAL_ERROR
          "Could not detect MUSA arch of installed GPUs. try specifying the TORCH_MUSA_ARCH_LIST env manually"
      )
    endif()
    set(_TMP ${MUSA_ARCH_INSTALLED})
  endif()
  set(${output_var} "${_TMP}")
endmacro()
