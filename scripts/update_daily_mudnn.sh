#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
mudnn_path=./daily_mudnn_${DATE}
#TODO:(lms) revise the mudnn version to daily 08.22 to avoid ci failure
mudnn_oss_link=https://oss.mthreads.com/release-ci/computeQA/ai/history/GPU_ARCH_MP_21/20230822muDNN20230822_developcompute_daily_pkg_full330/mudnn.tar
# mudnn_oss_link=http://oss.mthreads.com/release-ci/computeQA/ai/newest/mudnn.tar

wget --no-check-certificate ${mudnn_oss_link} -P ${mudnn_path}
tar -xvf ${mudnn_path}/mudnn.tar -C ${mudnn_path}
pushd ${mudnn_path}/mudnn
bash install_mudnn.sh
popd
