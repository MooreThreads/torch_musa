cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(example-app CXX C)

set(ENV{MUSA_HOME} "/usr/local/musa")

find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND
    ${Python_EXECUTABLE} -c "
import os
os.environ[\"TORCH_DEVICE_BACKEND_AUTOLOAD\"] = \"0\"
import torch, torch_musa
print(torch_musa.core.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_MUSA_PYTHONPATH)
string(REGEX REPLACE "^(.+)\n$" "\\1" TORCH_MUSA_PYTHONPATH
                     ${TORCH_MUSA_PYTHONPATH})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_MUSA_PYTHONPATH})
find_package(TorchMusa REQUIRED CONFIG)
list(POP_BACK CMAKE_PREFIX_PATH)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app torch_musa
                      "/usr/lib/x86_64-linux-gnu/libtinfo.so.6")
