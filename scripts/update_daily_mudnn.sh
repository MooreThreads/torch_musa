#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
mudnn_path=./daily_mudnn_${DATE}

wget --no-check-certificate http://oss.mthreads.com/release-ci/computeQA/ai/newest/mudnn.tar -P ${mudnn_path}
tar -xvf ${mudnn_path}/mudnn.tar -C ${mudnn_path}
pushd ${mudnn_path}/mudnn
bash install_mudnn.sh
popd
