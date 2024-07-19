#!/bin/bash  
versions=("38" "39" "310")  

MUSA_TOOLKITS_URL="https://oss.mthreads.com/release-rc/cuda_compatible/rc2.1.0/qy1/musa_toolkits_rc2.1.0-qy1.tar.gz"
MUDNN_URL="https://oss.mthreads.com/release-rc/cuda_compatible/rc2.1.0/qy1/mudnn_rc2.5.0-qy1.tar.gz"
MCCL_URL="https://oss.mthreads.com/release-rc/cuda_compatible/rc2.0.0/qy1/mccl_rc1.4.0-qy1.tar.gz"
TAG="rc2.1.0-v1.1.0-qy1"
WHL_URL="oss.mthreads.com/ai-product/torch_musa_release/v1.1.0-rc2/qy1" 

# MUSA_TOOLKITS_URL="https://oss.mthreads.com/release-rc/cuda_compatible/rc2.1.0/qy2/musa_toolkits_rc2.1.0-qy2.tar.gz"
# MUDNN_URL="https://oss.mthreads.com/release-rc/cuda_compatible/rc2.1.0/qy2/mudnn_rc2.5.0-qy2.tar.gz"
# MCCL_URL="https://oss.mthreads.com/release-rc/cuda_compatible/rc2.0.0/qy2/mccl_rc1.4.0-qy2.tar.gz"
# TAG="rc2.1.0-v1.1.0-qy2"
# WHL_URL="oss.mthreads.com/ai-product/torch_musa_release/v1.1.0-rc2/qy2" 

for version in "${versions[@]}"; do  
    command="bash build.sh -n sh-harbor.mthreads.com/mt-ai/musa-pytorch-release-py$version                  \
              -b ubuntu:20.04                                                                               \
              -f ./ubuntu/dockerfile.release                                                                \
              -v ${version:0:1}.${version:1}                                                                \
              -m ${MUSA_TOOLKITS_URL}                                                                       \
              -t ${TAG}                                                                                     \
              --mudnn_url ${MUDNN_URL}                                                                      \
              --mccl_url ${MCCL_URL}                                                                        \
              --torch_whl_url ${WHL_URL}/torch-2.0.0-cp$version-cp$version-linux_x86_64.whl           \
              --torch_musa_whl_url ${WHL_URL}/torch_musa-1.1.0-cp$version-cp$version-linux_x86_64.whl \
              --release"
    $command
    if [ $? -ne 0 ]; then
      echo -e "\033[31mFAILED TO BUILD! COMMAND:${command}  \033[0m"
      exit 1
    fi
done
