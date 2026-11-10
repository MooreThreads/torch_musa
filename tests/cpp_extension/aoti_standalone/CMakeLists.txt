cmake_minimum_required(VERSION 3.18)
project(musa_aoti_standalone LANGUAGES CXX)

set(USE_MCCL OFF)
find_package(TorchMusa CONFIG REQUIRED)

add_executable(aoti_standalone main.cpp)
target_compile_features(aoti_standalone PRIVATE cxx_std_20)
target_link_libraries(aoti_standalone PRIVATE torch_musa)

# This executable deliberately has no direct MUSA symbol references.
add_executable(backend_registration backend_registration.cpp)
target_link_libraries(backend_registration PRIVATE torch_musa)
