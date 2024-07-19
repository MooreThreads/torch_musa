#!/bin/bash
set -e
DATE=$(date +%Y%m%d)

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')

ARCH="qy1"

if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ] || [ "$GPU" = "MTTS80ES" ]; then
    ARCH="qy1"
elif [ "$GPU" = "MTTS4000" ] || [ "$GPU" = "MTTS90" ]; then
    ARCH="qy2"
else
  echo "Expect MTTS3000 | MTTS80 | MTTS4000 | MTTS90 | MTTS80ES, got ${GPU}"
fi

mudnn_path=./release_mudnn_${DATE}
MUDNN_VERSION=dev2.6.0
wget --no-check-certificate https://oss.mthreads.com/release-rc/cuda_compatible/dev3.0.0/${ARCH}/mudnn_${MUDNN_VERSION}-${ARCH}.tar.gz -P ${mudnn_path}
tar -zxvf ${mudnn_path}/mudnn_${MUDNN_VERSION}-${ARCH}.tar.gz -C ${mudnn_path}
pushd ${mudnn_path}/mudnn
bash install_mudnn.sh
popd
echo -e "\033[31mmudnn update to version ${MUDNN_VERSION}! \033[0m"
rm -rf ${mudnn_path}
