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
  echo_info "This script will install daily musa toolkits,"
  echo_info "Currently, the musa toolkits consist of the following components: musa_runtime, mcc, musify, muFFT, muBLAS, muPP, and muRAND."
}
help
DATE=$(date +%Y%m%d)

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')

ARCH="x86_64-ubuntu-mp_21"

if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ] || [ "$GPU" = "MTTS80ES" ]; then
    ARCH="x86_64-ubuntu-mp_21"
elif [ "$GPU" = "MTTS4000" ]; then
    ARCH="x86_64-ubuntu-mp_22"
else
    echo -e "\033[31mThe output of mthreads-gmi -q -i 0 | grep \"Product Name\" | awk -F: '{print \$2}' | tr -d '[:space:]' is not correct! Now GPU ARCH is set to qy1 by default! \033[0m"
fi

yesterday_date=$(TZ=Asia/Shanghai date -d "yesterday" +%Y%m%d)

echo -e "\033[31mWarning: update musa toolkit will uninstall mccl and mudnn! \033[0m"
rm -rf /usr/local/musa*
musa_toolkit_path=./daily_musa_toolkits_${DATE}
mkdir -p ${musa_toolkit_path}
wget --no-check-certificate https://oss.mthreads.com/release-ci/computeQA/cuda_compatible/${yesterday_date}/${ARCH}/musa_toolkits_install_full.tar.gz -P ${musa_toolkit_path}
tar -zxf ${musa_toolkit_path}/musa_toolkits_install_full.tar.gz -C ${musa_toolkit_path}
pushd ${musa_toolkit_path}/musa_toolkits_install
bash ./install.sh 
popd
echo -e "\033[31mmusa toolkits update to the newest version! \033[0m"
rm -rf ${musa_toolkit_path}
