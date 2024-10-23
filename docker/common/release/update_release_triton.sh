#!/bin/bash
set -e
DATE=$(date +%Y%m%d)

GPU=$(mthreads-gmi -i 0 -q | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')

if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ] || [ "$GPU" = "MTTS80ES" ]; then
  exit 0
fi

triton_path=./release_triton_${DATE}
read NAME TAG <<< `sed -n 1p $(dirname $0)/version.txt | awk -F: '{print $1, $2}'`
if [ $NAME != "MUSA" ]; then
  echo -e "\033[31mtriton load wrong musa version: $NAME:$TAG, check ./version.txt! \033[0m"
  exit 1
fi
read NAME VERSION <<< `sed -n 4p $(dirname $0)/version.txt | awk -F: '{print $1, $2}'`
if [ $NAME != "TRITON" ]; then
  echo -e "\033[31mload wrong triton version: $NAME:$VERSION, check ./version.txt! \033[0m"
  exit 1
fi

wget --no-check-certificate https://oss.mthreads.com/release-rc/cuda_compatible/${TAG}/triton_${VERSION}.tar.gz -P ${triton_path}
tar -zxvf ${triton_path}/triton_${VERSION}.tar.gz -C ${triton_path}
pushd ${triton_path}
pip install triton-*.whl
popd
echo -e "\033[31mtriton update to version ${TAG}-${VERSION}! \033[0m"
rm -rf ${triton_path}
