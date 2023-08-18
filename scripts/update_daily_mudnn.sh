#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
# TODO:(mtai) temporarily change mudnn to daily 08.15 version. mudnn after that day will cause many problems.
# Those problems are issued in: https://github.mthreads.com/mthreads/muDNN/issues/304
# When it is done, we should change the version back to the newest.
mudnn_path=./daily_mudnn_${DATE}

mudnn_oss_link=https://oss.mthreads.com/release-ci/computeQA/ai/history/GPU_ARCH_MP_21/20230815muDNN20230815_developcompute_daily_pkg_full320/mudnn.tar
# mudnn_oss_link=http://oss.mthreads.com/release-ci/computeQA/ai/newest/mudnn.tar

wget --no-check-certificate ${mudnn_oss_link} -P ${mudnn_path}
tar -xvf ${mudnn_path}/mudnn.tar -C ${mudnn_path}
pushd ${mudnn_path}/mudnn
bash install_mudnn.sh
popd
