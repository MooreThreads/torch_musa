#!/bin/bash
set -e
DATE=$(date +%Y%m%d)

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')
ARCH="cc3.1"
if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ] || [ "$GPU" = "MTTS80ES" ] || [ "$GPU" = "MTTS4000" ]; then
    ARCH="cc2.2"
fi

mudnn_path=./release_mudnn_${DATE}
read MUSA TAG <<< `sed -n 1p $(dirname $0)/version.txt | awk -F: '{print $1, $2}'`
if [ "$MUSA" != "MUSA" ]; then
  echo -e "\033[31mmudnn load wrong musa version: $MUSA:$TAG, check ./version.txt! \033[0m"
  exit 1
fi
read NAME MUDNN_VERSION <<< `sed -n 3p $(dirname $0)/version.txt | awk -F: '{print $1, $2}'`
if [ "$NAME" != "MUDNN" ]; then
  echo -e "\033[31mload wrong mudnn version: $NAME:$MUDNN_VERSION, check ./version.txt! \033[0m"
  exit 1
fi

wget --no-check-certificate https://oss.mthreads.com/release-rc/cuda_compatible/${TAG}/${ARCH}/mudnn_${MUDNN_VERSION}.${ARCH}.tar.gz -P ${mudnn_path}
tar -zxvf ${mudnn_path}/mudnn_${MUDNN_VERSION}.${ARCH}.tar.gz -C ${mudnn_path}
pushd ${mudnn_path}/mudnn
bash install_mudnn.sh
popd
echo -e "\033[31mmudnn update to version ${MUDNN_VERSION}! \033[0m"
rm -rf ${mudnn_path}
