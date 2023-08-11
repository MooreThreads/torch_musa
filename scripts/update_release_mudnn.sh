#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
mudnn_path=./release_mudnn_${DATE}
wget --no-check-certificate https://oss.mthreads.com/release-rc/cuda_compatible/rc1.4.0/mudnn_rtm2.1.1.tar -P ${mudnn_path}
tar -xvf ${mudnn_path}/mudnn_rtm2.1.1.tar -C ${mudnn_path}
pushd ${mudnn_path}/mudnn
bash install_mudnn.sh
popd
