#!/bin/bash
scripts_dir="$(dirname "$(realpath "${BASH_SOURCE:-$0}")")"
bash ${scripts_dir}/update_release_musa_toolkits.sh
bash ${scripts_dir}/update_release_mccl.sh
bash ${scripts_dir}/update_release_mudnn.sh
bash ${scripts_dir}/update_release_mupti.sh
bash ${scripts_dir}/update_release_math.sh -w

