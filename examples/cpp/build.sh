#!/bin/bash
if [ -d "./build" ]; then
  rm -r build
fi
mkdir -p build
pushd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch_musa; print(torch_musa.core.cmake_prefix_path)')" ..
make -j 10
popd
