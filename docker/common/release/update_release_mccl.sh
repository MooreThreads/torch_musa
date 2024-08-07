#!/bin/bash
set -e
scripts_dir="$(dirname "$(realpath "${BASH_SOURCE:-$0}")")"
torch_musa_dir="$(dirname "$scripts_dir")"

DATE=$(date +%Y%m%d)

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')

ARCH="qy1"

if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ] || [ "$GPU" = "MTTS80ES" ]; then
    ARCH="qy1"
elif [ "$GPU" = "MTTS4000" ] || [ "$GPU" = "MTTS90" ]; then
    ARCH="qy2"
else
    echo "Expect MTTS3000 | MTTS80 | MTTS4000 | MTTS90, got ${GPU}"
fi

if [ "$ARCH" = "qy1" ]; then
    # mccl qy1 dev3.0.0 not released
    bash ${torch_musa_dir}/install_mccl.sh --mccl_url "https://oss.mthreads.com/release-rc/cuda_compatible/rc2.0.0/${ARCH}/mccl_rc1.4.0-${ARCH}.tar.gz"
elif [ "$ARCH" = "qy2" ]; then
    bash ${torch_musa_dir}/install_mccl.sh --mccl_url "https://oss.mthreads.com/release-rc/cuda_compatible/dev3.0.0/${ARCH}/mccl_dev1.6.0-${ARCH}.tar.gz"
fi
echo -e "\033[31mmccl update to version qy1 rc1.4.0/qy2 dev1.6.0! \033[0m"
