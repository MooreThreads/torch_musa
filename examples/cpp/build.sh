#!/bin/bash
if [ -d "./build" ]; then
  rm -r build
fi
mkdir -p build
pushd build
PY_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d. -f1-2)
export PY_VERSION=$PY_VERSION
cmake -DPython_VERSION=$PY_VERSION -DCMAKE_PREFIX_PATH="$(python -c 'import torch_musa; print(torch_musa.core.cmake_prefix_path)')" ..
make -j 10
popd
