#!/bin/bash
#Currently, the musa toolkits consist of the following components: musa_runtime, mcc, musify, muFFT, muBLAS, muPP, and muRAND.
set -e

echo_info() {
  echo -e "\033[33m"$1"\033[0m"
}

echo_success() {
  echo -e "\033[32m"$1"\033[0m"
}

help() {
  echo_info "----------------------------------------------------------------"
  name="$(basename "$(realpath "${BASH_SOURCE:-$0}")")"
  echo_info "Description:"
  echo_info "This script will install release musa toolkits,"
  echo_info "Currently, the musa toolkits consist of the following components: musa_runtime, mcc, musify, muFFT, muBLAS, muPP, and muRAND."
}
help

DATE=$(date +%Y%m%d)

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')
ARCH="cc3.1"
if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ] || [ "$GPU" = "MTTS80ES" ] || [ "$GPU" = "MTTS4000" ]; then
    ARCH="cc2.2"
fi

read NAME TAG <<< `sed -n 1p $(dirname $0)/version.txt | awk -F: '{print $1, $2}'`
if [ "$NAME" != "MUSA" ]; then
  echo -e "\033[31mmusa_toolkit load wrong musa version: $NAME:$TAG, check ./version.txt! \033[0m"
  exit 1
fi

echo -e "\033[31mWarning: update musa toolkit will uninstall mccl and mudnn! \033[0m"
rm -rf /usr/local/musa*
musa_toolkit_path=./release_musa_toolkits_${DATE}
mkdir -p ${musa_toolkit_path}
wget --no-check-certificate https://oss.mthreads.com/release-rc/cuda_compatible/${TAG}/${ARCH}/musa_toolkits_${TAG}.tar.gz -P ${musa_toolkit_path}

tar -zxvf ${musa_toolkit_path}/musa_toolkits_${TAG}.tar.gz -C ${musa_toolkit_path}
pushd ${musa_toolkit_path}/musa_toolkits_install
bash ./install.sh 
popd

echo -e "\033[31mmusa toolkits update to version ${TAG}! \033[0m"
rm -rf ${musa_toolkit_path}
