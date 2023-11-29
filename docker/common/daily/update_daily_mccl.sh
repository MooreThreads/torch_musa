#!/bin/bash
set -e

scripts_dir="$(dirname "$(realpath "${BASH_SOURCE:-$0}")")"
torch_musa_dir="$(dirname "$scripts_dir")"
bash ${torch_musa_dir}/install_mccl.sh --mccl_url "https://oss.mthreads.com/release-ci/computeQA/mathX/newest/mccl.tar.gz"

echo -e "\033[31mmccl update to the newest version! \033[0m"