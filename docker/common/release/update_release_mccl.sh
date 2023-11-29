#!/bin/bash
set -e
scripts_dir="$(dirname "$(realpath "${BASH_SOURCE:-$0}")")"
torch_musa_dir="$(dirname "$scripts_dir")"

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

bash ${torch_musa_dir}/install_mccl.sh --mccl_url "https://oss.mthreads.com/release-rc/cuda_compatible/dev1.5.1/${ARCH}/mccl_dev1.3.0.tar"

echo -e "\033[31mmccl update to version dev1.3.0! \033[0m"