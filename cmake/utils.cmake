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
    MUSA_RUNTIME_VERSION_EXECUTABLE
    NAMES musa_runtime_version
    PATHS "$ENV{MUSA_HOME}"
    PATH_SUFFIXES bin bin64
    NO_DEFAULT_PATH)
  mark_as_advanced(MUSA_RUNTIME_VERSION_EXECUTABLE)
  if(MUSA_RUNTIME_VERSION_EXECUTABLE)
    execute_process(COMMAND ${MUSA_RUNTIME_VERSION_EXECUTABLE}
                    OUTPUT_VARIABLE MUSA_RUNTIME_VERSION)
    string(REGEX REPLACE ".*\"([0-9]+)\\.([0-9]+)\\.([0-9]+)\".*" "\\1"
                         MUSA_RUNTIME_VERSION_MAJOR ${MUSA_RUNTIME_VERSION})
    string(REGEX REPLACE ".*\"([0-9]+)\\.([0-9]+)\\.([0-9]+)\".*" "\\2"
                         MUSA_RUNTIME_VERSION_MINOR ${MUSA_RUNTIME_VERSION})
    string(REGEX REPLACE ".*\"([0-9]+)\\.([0-9]+)\\.([0-9]+)\".*" "\\3"
                         MUSA_RUNTIME_VERSION_PATCH ${MUSA_RUNTIME_VERSION})

    math(
      EXPR
      MUSA_RUNTIME_VERSION_INT
      "${MUSA_RUNTIME_VERSION_MAJOR} * 1000 + ${MUSA_RUNTIME_VERSION_MINOR} * 10 + ${MUSA_RUNTIME_VERSION_PATCH}"
    )
    set(${outputvar} "${MUSA_RUNTIME_VERSION_INT}" PARENT_SCOPE)
    mark_as_advanced(${outputvar})
  endif()
endfunction()

# Return the MUSA arch flags, if not specified by TORCH_MUSA_ARCH_LIST, then
# auto detect the arch of installed GPUs. Usage:
# torch_musa_get_mcc_arch_list(output_var)
macro(torch_musa_get_mcc_arch_list output_var)
  set(_MUSA_GPU_ARCHS)
  if(DEFINED ENV{TORCH_MUSA_ARCH_LIST})
    set(_MUSA_GPU_ARCHS $ENV{TORCH_MUSA_ARCH_LIST})
  else()
    # Auto detect the arch of installed GPUs
    set(_TMP_FILE "${PROJECT_BINARY_DIR}/get_musa_compute_capabilities.cpp")
    file(
      WRITE ${_TMP_FILE}
      ""
      "#include <musa_runtime.h>\n"
      "#include <cstdio>\n"
      "int main()\n"
      "{\n"
      "  int dev_count = 0;\n"
      "  if (musaSuccess != musaGetDeviceCount(&dev_count)) return -1;\n"
      "  if (dev_count == 0) return -1;\n"
      "  for (int device = 0; device < dev_count; ++device)\n"
      "  {\n"
      "    musaDeviceProp prop;\n"
      "    if (musaSuccess == musaGetDeviceProperties(&prop, device))\n"
      "      std::printf(\"%d%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")

    try_run(
      RUN_RESULT COMPILE_RESULT ${PROJECT_BINARY_DIR} ${_TMP_FILE}
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${MUSA_INCLUDE_DIRS}" LINK_LIBRARIES
                  ${MUSA_LIBRARIES} RUN_OUTPUT_VARIABLE musa_cc)
    if(RUN_RESULT EQUAL 0)
      string(REPLACE " " ";" _tmp_musa_cc_list "${musa_cc}")
      list(REMOVE_DUPLICATES _tmp_musa_cc_list)
      set(_MUSA_GPU_ARCHS ${_tmp_musa_cc_list})
    endif()
    unset(_TMP_FILE)
    unset(_tmp_musa_cc_list)
    unset(musa_cc)
  endif()

  if(NOT _MUSA_GPU_ARCHS)
    message(
      FATAL_ERROR
        "Could not detect MUSA arch of installed GPUs. try specifying the TORCH_MUSA_ARCH_LIST env manually"
    )
  endif()
  set(${output_var} "${_MUSA_GPU_ARCHS}")
  unset(_MUSA_GPU_ARCHS)
endmacro()
