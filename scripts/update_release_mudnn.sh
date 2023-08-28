#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
mudnn_path=./release_mudnn_${DATE}
wget --no-check-certificate https://oss.mthreads.com/mt-ai-data/infra/framework/torch_musa/dependency/mudnn/mudnn.tar -P ${mudnn_path}
tar -xf ${mudnn_path}/mudnn.tar -C ${mudnn_path}
pushd ${mudnn_path}/mudnn
bash install_mudnn.sh
popd
