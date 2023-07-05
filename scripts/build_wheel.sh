#!/bin/bash

set -exo pipefail

if [ -z "$PYTORCH_REPO_PATH" ]; then
  PYTORCH_REPO_PATH="/home/pytorch"
fi

bash build.sh -w
