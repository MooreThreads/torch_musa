#!/bin/bash

set -exo pipefail

if [ -z "$PYTORCH_REPO_PATH" ]; then
  PYTORCH_REPO_PATH="/home/pytorch"
fi

rm -rf build $PYTORCH_REPO_PATH/build
rm -rf dist $PYTORCH_REPO_PATH/dist
bash build.sh -w
