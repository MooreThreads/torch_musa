#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
mudnn_path=./daily_mudnn_${DATE}

wget --no-check-certificate https://oss.mthreads.com/release-rc/computeQA/cuda_compatible/rc1.4.0-rc1/mudnn.tar -P ${mudnn_path}
tar -xvf ${mudnn_path}/mudnn.tar -C ${mudnn_path}
pushd ${mudnn_path}/mudnn
bash install_mudnn.sh
popd
