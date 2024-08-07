#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
mudnn_path=./daily_mudnn_${DATE}

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')

ARCH="x86_64-ubuntu-mp_21"

if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ] || [ "$GPU" = "MTTS80ES" ]; then
    ARCH="x86_64-ubuntu-mp_21"
elif [ "$GPU" = "MTTS4000" ] || [ "$GPU" = "MTTS90" ]; then
    ARCH="x86_64-ubuntu-mp_22"
else
    echo -e "\033[31mThe output of mthreads-gmi -q -i 0 | grep \"Product Name\" | awk -F: '{print \$2}' | tr -d '[:space:]' is not correct! Now GPU ARCH is set to qy1 by default! \033[0m"
fi

yesterday_date=$(TZ=Asia/Shanghai date -d "yesterday" +%Y%m%d)

mudnn_oss_link="https://oss.mthreads.com/release-ci/computeQA/cuda_compatible/${yesterday_date}/${ARCH}/mudnn.tar.gz"

wget --no-check-certificate ${mudnn_oss_link} -P ${mudnn_path}
tar -xvzf ${mudnn_path}/mudnn.tar.gz -C ${mudnn_path}
pushd ${mudnn_path}/mudnn
bash install_mudnn.sh
popd
echo -e "\033[31mmudnn update to the newest version! \033[0m"
rm -rf ${mudnn_path}