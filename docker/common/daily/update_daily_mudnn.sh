#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
mudnn_path=./daily_mudnn_${DATE}

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')
ARCH="cc3.1"
if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ] || [ "$GPU" = "MTTS80ES" ] || [ "$GPU" = "MTTS4000" ]; then
    ARCH="cc2.2"
fi

yesterday_date=$(TZ=Asia/Shanghai date -d "yesterday" +%Y-%m-%d)
oss_link="https://oss.mthreads.com/release-ci/computeQA/cuda_compatible/release_KUAE_2.0_for_PH1_M3D/${yesterday_date}"

if [ "${ARCH}" = "cc3.1" ]; then
    mudnn_oss_link="${oss_link}/mudnn_dev2.8.0.PH1.tar.gz"
else
    mudnn_oss_link="${oss_link}/mudnn_dev2.8.0.tar.gz"
fi

wget --no-check-certificate ${mudnn_oss_link} -P ${mudnn_path}
tar -xvzf ${mudnn_path}/mudnn.tar.gz -C ${mudnn_path}
pushd ${mudnn_path}/mudnn
bash install_mudnn.sh
popd
echo -e "\033[31mmudnn update to the newest version! \033[0m"
rm -rf ${mudnn_path}
