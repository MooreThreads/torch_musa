#!/bin/bash
# TODO: update torchaudio, torchvision versions
versions=("310")  

SW_TAG=dev4.0.0
MUDNN_VERSION=dev2.8.0
MCCL_VERSION=dev1.8.0
TORCH_MUSA_VERSION=1.3.2
OSS_PREFIX=https://oss.mthreads.com/release-rc/cuda_compatible

WHL_URL="oss.mthreads.com/ai-product/release-rc/torch_musa/rc${TORCH_MUSA_VERSION}"

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')
ARCH="ph1"
if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ] || [ "$GPU" = "MTTS80ES" ]; then
  ARCH="qy1"
elif [ "$GPU" = "MTTS4000" ]; then
  ARCH="qy2"
fi

for version in "${versions[@]}"; do
  if [ "$ARCH" = "qy1" ] || ["$ARCH" = "qy2" ];then
    CC=cc2.2
  else
    CC=cc3.1
  fi
  MUSA_TOOLKITS_URL="${OSS_PREFIX}/${SW_TAG}/${CC}/musa_toolkits_${SW_TAG}.tar.gz"
  MUDNN_URL="${OSS_PREFIX}/${SW_TAG}/${CC}/mudnn_${MUDNN_VERSION}.${CC}.tar.gz"
  MCCL_URL="${OSS_PREFIX}/${SW_TAG}/${CC}/mccl_${MCCL_VERSION}.${CC}.tar.gz"
  TAG="${SW_TAG}-v${TORCH_MUSA_VERSION}-${ARCH}"
  command="bash build.sh -n sh-harbor.mthreads.com/mt-ai/musa-pytorch-release-py$version                                    \
            -b ubuntu:22.04                                                                                                 \
            -f ./ubuntu/dockerfile.release                                                                                  \
            -v ${version:0:1}.${version:1}                                                                                  \
            -m ${MUSA_TOOLKITS_URL}                                                                                         \
            -t ${TAG}                                                                                                       \
            --mudnn_url ${MUDNN_URL}                                                                                        \
            --mccl_url ${MCCL_URL}                                                                                          \
            --torch_whl_url ${WHL_URL}/torch-2.2.0-cp$version-cp$version-linux_x86_64.whl                                   \
            --torch_musa_whl_url ${WHL_URL}/${ARCH}/torch_musa-${TORCH_MUSA_VERSION}-cp$version-cp$version-linux_x86_64.whl \
            --torchaudio_whl_url ${WHL_URL}/2.2.2+cefdb36-cp$version-cp$version-linux_x86_64.whl                            \
            --torchvision_whl_url ${WHL_URL}/0.17.2+c1d70fe-cp$version-cp$version-linux_x86_64.whl                          \
            --release"
  $command
  if [ $? -ne 0 ]; then
    echo -e "\033[31mFAILED TO BUILD! COMMAND:${command}  \033[0m"
    exit 1
  fi
done
