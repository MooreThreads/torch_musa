#!/bin/bash
set -e

scripts_dir="$(dirname "$(realpath "${BASH_SOURCE:-$0}")")"
torch_musa_dir="$(dirname "$scripts_dir")"

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')
ARCH="ph1"
if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ] || [ "$GPU" = "MTTS80ES" ]; then
    ARCH="qy1"
elif [ "$GPU" = "MTTS4000" ]; then
    ARCH="qy2"
fi

yesterday_date=$(TZ=Asia/Shanghai date -d "yesterday" +%Y-%m-%d)
oss_link="https://oss.mthreads.com/release-ci/computeQA/cuda_compatible/CI/release_KUAE_2.0_for_PH1_M3D/${yesterday_date}"

if [ "${ARCH}" = "qy1" ]; then
    mccl_oss_link="${oss_link}/mccl_dev1.8.0.tar.gz"
elif [ "$ARCH" = "qy2" ]; then
    mccl_oss_link="${oss_link}/mccl_dev1.8.0.tar.gz"
else
    mccl_oss_link="${oss_link}/mccl_dev1.8.0.PH1.tar.gz"
fi

bash ${torch_musa_dir}/install_mccl.sh --mccl_url ${mccl_oss_link}

echo -e "\033[31mmccl update to the newest version! \033[0m"
