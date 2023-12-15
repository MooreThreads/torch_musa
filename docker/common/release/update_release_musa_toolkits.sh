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

ARCH="qy1"
if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ]; then
    ARCH="qy1"
elif [ "$GPU" = "MTTS4000" ] || [ "$GPU" = "MTTS90" ]; then
    ARCH="qy2"
else
    echo -e "\033[31mThe output of mthreads-gmi -q -i 0 | grep \"Product Name\" | awk -F: '{print \$2}' | tr -d '[:space:]' is not correct! Now GPU ARCH is set to qy1 by default! \033[0m"
fi

musa_toolkit_path=./release_musa_toolkits_${DATE}
mkdir -p ${musa_toolkit_path}
wget --no-check-certificate https://oss.mthreads.com/release-rc/cuda_compatible/dev1.5.1/${ARCH}/musa_toolkits_dev1.5.1-${ARCH}.tar.gz -P ${musa_toolkit_path}
tar -zxf ${musa_toolkit_path}/musa_toolkits_dev1.5.1-${ARCH}.tar.gz -C ${musa_toolkit_path}
pushd ${musa_toolkit_path}/musa_toolkits_install
bash ./install.sh 
popd

echo -e "\033[31mmusa toolkits update to version dev1.5.1! \033[0m"
