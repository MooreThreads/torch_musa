#!/bin/bash
set -e
scripts_dir="$(dirname "$(realpath "${BASH_SOURCE:-$0}")")"
torch_musa_dir="$(dirname "$scripts_dir")"

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')
ARCH="cc3.1"
if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ] || [ "$GPU" = "MTTS80ES" ] || [ "$GPU" = "MTTS4000" ]; then
    ARCH="cc2.2"
fi

read MUSA TAG <<< `sed -n 1p $(dirname $0)/version.txt | awk -F: '{print $1, $2}'`
if [ "$MUSA" != "MUSA" ]; then
  echo -e "\033[31mmccl load wrong musa version: $MUSA:$TAG, check ./version.txt! \033[0m"
  exit 1
fi
read NAME VERSION <<< `sed -n 2p $(dirname $0)/version.txt | awk -F: '{print $1, $2}'`
if [ "$NAME" != "MCCL" ]; then
 . echo -e "\033[31mload wrong mccl version: $NAME:$VERSION, check ./version.txt! \033[0m"
  exit 1
fi

OSS_PREFIX=https://oss.mthreads.com/release-ci/computeQA/cuda_compatible/CI/release_KUAE_2.0_for_PH1_M3D/2025-04-08
MCCL_VERSION=dev1.8.0
MCCL_URL="${OSS_PREFIX}/mccl_${MCCL_VERSION}.tar.gz"

bash ${torch_musa_dir}/install_mccl.sh --mccl_url ${MCCL_URL}
echo -e "\033[31mmccl update to version ${TAG}-${VERSION}! \033[0m"
