#!/bin/bash
# Use this script to check that the torch_musa inside the newly
# built docker image is compatiple with the corresponding musa software.
# Example: bash scripts/docker/auto_check_new_image_is_available.sh sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev:latest

DOCKER_IMAGE_NAME=$1
DOCKER_CONTAINER_NAME="torch_musa_ut_check_"$(date +%s)

torch_musa_dir="/home/torch_musa"
exec_command_prefix="docker exec -i ${DOCKER_CONTAINER_NAME} bash -c "
ret_code_file="/tmp/ret_code"
check_ret_code_cmd="${exec_command_prefix} \" cat ${ret_code_file}\""

# start a docker container
docker run --rm --init --detach --privileged --network=host --name=${DOCKER_CONTAINER_NAME} --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g ${DOCKER_IMAGE_NAME} sleep infinity

# build and install torch_musa
${exec_command_prefix} "cd ${torch_musa_dir} && bash build.sh -m 2>&1 | tee build.log && echo \${PIPESTATUS[0]} > ${ret_code_file}"
if [ "$(eval ${check_ret_code_cmd})" -ne 0 ]; then
  echo "Build torch_musa failed, check build log (${exec_command_prefix} \"cat ${torch_musa_dir}/build.log\") for more details."
  exit 1
fi

# check unit test
${exec_command_prefix} "cd ${torch_musa_dir} && bash scripts/run_unittest.sh 2>&1 | tee run_unittest.log && echo \${PIPESTATUS[0]} > ${ret_code_file}"
if [ "$(eval ${check_ret_code_cmd})" -ne 0 ]; then
  echo "Torch_musa ut failed, check unittest log (${exec_command_prefix} \"cat ${torch_musa_dir}/run_unittest.log\") for more details."
  exit 1
fi

echo "==============================Check passed!=============================="

docker container stop ${DOCKER_CONTAINER_NAME}
