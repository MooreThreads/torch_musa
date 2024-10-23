#!/bin/bash
scripts_dir="$(dirname "$(realpath "${BASH_SOURCE:-$0}")")"
bash ${scripts_dir}/update_daily_musa_toolkits.sh
bash ${scripts_dir}/update_daily_mccl.sh
bash ${scripts_dir}/update_daily_mudnn.sh
bash ${scripts_dir}/update_daily_mupti.sh
bash ${scripts_dir}/update_daily_alg_thrust.sh
