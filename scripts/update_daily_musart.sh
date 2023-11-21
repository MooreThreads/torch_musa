#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
scripts_dir="$(dirname "$(realpath "${BASH_SOURCE:-$0}")")"
torch_musa_dir="$(dirname "$scripts_dir")"

####### musa toolkits install, this needs latest stable DDK.
# uninstall previous musart first
pushd /usr/local
rm -rf musa musa-*
popd

bash ${torch_musa_dir}/scripts/env_update.sh -t -r -c
####### end

####### install math components
bash ${torch_musa_dir}/docker/common/install_math.sh -w
####### end

####### install mccl
bash ${torch_musa_dir}/docker/common/install_mccl.sh --mccl_url "https://oss.mthreads.com/release-ci/computeQA/mathX/newest/mccl.tar.gz"
####### end

###### install muBLAS (this is excluded by the install_math.sh)
