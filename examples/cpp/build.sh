#!/bin/bash
if [ -d "./build" ]; then
  rm -r build
fi
mkdir -p build

PY=$(which python 2>/dev/null || which python3 2>/dev/null)

export TORCH_DEVICE_BACKEND_AUTOLOAD=0

pushd build
cmake ..
make -j 10
popd
