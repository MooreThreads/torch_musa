#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
scripts_dir="$(dirname "$(realpath "${BASH_SOURCE:-$0}")")"
torch_musa_dir="$(dirname "$scripts_dir")"

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')
ARCH="GPU_ARCH_MP_21"
if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ]; then
    ARCH="GPU_ARCH_MP_21"
elif [ "$GPU" = "MTTS4000" ] || [ "$GPU" = "MTTS90" ]; then
    ARCH="GPU_ARCH_MP_22"
else
    echo -e "\033[31mThe output of mthreads-gmi -q -i 0 | grep \"Product Name\" | awk -F: '{print \$2}' | tr -d '[:space:]' is not correct! Now GPU ARCH is set to qy1 by default! \033[0m"
fi

####### musa toolkits install, this needs latest stable DDK.

# uninstall previous musart first
pushd /usr/local
rm -rf musa musa-*
popd

bash ${torch_musa_dir}/env_update.sh -t -r -c
####### end

####### install math components
bash ${torch_musa_dir}/install_math.sh -w
####### end

####### install mccl
bash ${torch_musa_dir}/install_mccl.sh --mccl_url "https://oss.mthreads.com/release-ci/computeQA/mathX/newest/mccl.tar.gz"
####### end

###### install muBLAS (this is excluded by the install_math.sh)
