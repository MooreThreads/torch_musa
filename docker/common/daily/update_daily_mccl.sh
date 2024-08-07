#!/bin/bash
set -e

scripts_dir="$(dirname "$(realpath "${BASH_SOURCE:-$0}")")"
torch_musa_dir="$(dirname "$scripts_dir")"

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

bash ${torch_musa_dir}/install_mccl.sh --mccl_url "https://oss.mthreads.com/release-ci/computeQA/cuda_compatible/${yesterday_date}/${ARCH}/mccl.tar.gz"

echo -e "\033[31mmccl update to the newest version! \033[0m"
