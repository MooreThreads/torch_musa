cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(example-app CXX C)

find_package(TorchMusa REQUIRED)

add_executable(example-app example-app.cpp)

set_target_properties(example-app PROPERTIES CXX_STANDARD 17 LINK_FLAGS "-Wl,--no-as-needed")
# libmusa.so need to link the specified libtinfo.so.
target_link_libraries(example-app ${TorchMusa_LIBRARIES} /usr/lib/x86_64-linux-gnu/libtinfo.so.6)
