# Prints Torch_MUSA building configuration summary
function(torch_musa_build_configuration_summary)
  message(STATUS "CMake version           : ${CMAKE_VERSION}")
  message(STATUS "CMake command           : ${CMAKE_COMMAND}")
  message(STATUS "System                  : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "C++ compiler            : ${CMAKE_CXX_COMPILER}")
  message(STATUS "C++ compiler id         : ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "C++ compiler version    : ${CMAKE_CXX_COMPILER_VERSION}")
  message(STATUS "CXX flags               : ${CMAKE_CXX_FLAGS}")
  get_directory_property(tmp DIRECTORY ${PROJECT_SOURCE_DIR}
                                       COMPILE_DEFINITIONS)
  message(STATUS "CMAKE_PREFIX_PATH       : ${CMAKE_PREFIX_PATH}")
  message(STATUS "CMAKE_INSTALL_PREFIX    : ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "USE_PYTHON              : ${USE_PYTHON}")
  message(STATUS "BUILD_TEST              : ${BUILD_TEST}")
  message(STATUS "BUILD_TYPE              : ${CMAKE_BUILD_TYPE}")
  message(STATUS "PYTORCH_SOURCE_PATH     : $ENV{PYTORCH_REPO_PATH}")
  message(
    STATUS
      "PYTORCH_HEADERS_PATH    : ${CMAKE_BINARY_DIR}/${GENERATED_PORTING_DIR}/include"
  )
  message(STATUS "TORCH_MUSA_ARCH_LIST    : ${TORCH_MUSA_ARCH_LIST}")
  message(STATUS "MUSA TOOLKITS PATH      : ${MUSA_TOOLKIT_ROOT_DIR}")
  message(STATUS "MUSAToolkits_LIBRARIES  : ${MUSA_LIBRARIES}")
  message(STATUS "REAL_MUSA_VERSION       : ${REAL_MUSA_VERSION}")
  message(STATUS "MUDNN_LIBRARIES         : ${MUDNN_LIBRARIES}")

endfunction()
