#!/bin/bash
# We will update mudnn in Stable testing mainly in two cases:
# 1. mudnn changed its interfaces only, no need to update docker image
# 2. mudnn's dependencies changed, such as musa runtime, we'd better update docker image
set -e

DATE=$(date +%Y%m%d)
mudnn_path=./release_mudnn_${DATE}
wget --no-check-certificate https://oss.mthreads.com/mt-ai-data/infra/framework/torch_musa/dependency/mudnn/mudnn.tar -P ${mudnn_path}
tar -xf ${mudnn_path}/mudnn.tar -C ${mudnn_path}
pushd ${mudnn_path}/mudnn
bash install_mudnn.sh
popd
