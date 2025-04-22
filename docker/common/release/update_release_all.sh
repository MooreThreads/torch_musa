#!/bin/bash
set -ex
scripts_dir="$(dirname "$(realpath "${BASH_SOURCE:-$0}")")"
bash ${scripts_dir}/update_release_musa_toolkits.sh
bash ${scripts_dir}/update_release_mccl.sh
bash ${scripts_dir}/update_release_mudnn.sh
# bash ${scripts_dir}/update_release_triton.sh  # triton release TBA
bash ${scripts_dir}/update_release_alg_thrust.sh
