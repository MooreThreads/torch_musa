#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
mudnn_path=./daily_mudnn_${DATE}

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')
ARCH="GPU_ARCH_MP_21"
if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ]; then
    ARCH="GPU_ARCH_MP_21"
elif [ "$GPU" = "MTTS4000" ] || [ "$GPU" = "MTTS90" ]; then
    ARCH="GPU_ARCH_MP_22"
else
    echo -e "\033[31mThe output of mthreads-gmi -q -i 0 | grep \"Product Name\" | awk -F: '{print \$2}' | tr -d '[:space:]' is not correct! Now GPU ARCH is set to qy1 by default! \033[0m"
fi


#TODO:(lms) revise the mudnn version to daily 08.22 to avoid ci failure
# mudnn_oss_link=https://oss.mthreads.com/release-ci/computeQA/ai/history/GPU_ARCH_MP_21/20230822muDNN20230822_developcompute_daily_pkg_full330/mudnn.tar
mudnn_oss_link="http://oss.mthreads.com/release-ci/computeQA/ai/newest/${ARCH}/mudnn.tar"

wget --no-check-certificate ${mudnn_oss_link} -P ${mudnn_path}
tar -xvf ${mudnn_path}/mudnn.tar -C ${mudnn_path}
pushd ${mudnn_path}/mudnn
bash install_mudnn.sh
popd
echo -e "\033[31mmudnn update to the newest version! \033[0m"