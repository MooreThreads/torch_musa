#!/bin/bash
set -e
DATE=$(date +%Y%m%d)

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')

ARCH="qy1"

if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ]; then
    ARCH="qy1"
elif [ "$GPU" = "MTTS4000" ] || [ "$GPU" = "MTTS90" ]; then
    ARCH="qy2"
else
    echo -e "\033[31mThe output of mthreads-gmi -q -i 0 | grep \"Product Name\" | awk -F: '{print \$2}' | tr -d '[:space:]' is not correct! Now GPU ARCH is set to qy1 by default! \033[0m"
fi

mudnn_path=./release_mudnn_${DATE}
wget --no-check-certificate https://oss.mthreads.com/release-rc/cuda_compatible/rc2.0.0/${ARCH}/mudnn_rc2.4.0-${ARCH}.tar.gz -P ${mudnn_path}
tar -zxvf ${mudnn_path}/mudnn_rc2.4.0-${ARCH}.tar.gz -C ${mudnn_path}
pushd ${mudnn_path}/mudnn
bash install_mudnn.sh
popd
echo -e "\033[31mmudnn update to version rc2.4.0! \033[0m"
rm -rf ${mudnn_path}
