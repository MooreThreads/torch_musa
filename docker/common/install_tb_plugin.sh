#!/bin/bash
set -ex

PYTORCH_PATH=${PYTORCH_REPO_PATH:-/home/pytorch}

install_tb_plugin() {
  pip list | grep torch-tb-profiler
  if [ $? -ne 0 ]; then
    echo -e "\033[31mInstalling tb_plugin... \033[0m"
    pushd ${PYTORCH_PATH}/third_party/kineto/tb_plugin
    python setup.py sdist bdist_wheel
    pip install dist/torch_tb_profiler*.whl
    # to be compatible with tensorboard
    pip install protobuf==4.25
    popd
  fi
}

install_tb_plugin
