#!/bin/bash
set -e
scripts_dir="$(dirname "$(realpath "${BASH_SOURCE:-$0}")")"
torch_musa_dir="$(dirname "$scripts_dir")"

read MUSA TAG <<< `sed -n 1p $(dirname $0)/version.txt | awk -F: '{print $1, $2}'`
if [ $MUSA != "MUSA" ]; then
  echo -e "\033[31mmccl load wrong musa version: $MUSA:$TAG, check ./version.txt! \033[0m"
  exit 1
fi
read NAME VERSION <<< `sed -n 2p $(dirname $0)/version.txt | awk -F: '{print $1, $2}'`
if [ $NAME != "MCCL" ]; then
 . echo -e "\033[31mload wrong mccl version: $NAME:$VERSION, check ./version.txt! \033[0m"
  exit 1
fi

bash ${torch_musa_dir}/install_mccl.sh --mccl_url "https://oss.mthreads.com/release-rc/cuda_compatible/${TAG}/mccl_${VERSION}.tar.gz"
echo -e "\033[31mmccl update to version ${TAG}-${VERSION}! \033[0m"
