#!/bin/bash
# TODO: update torchaudio, torchvision versions
versions=("38" "39" "310")  
archs=("qy1" "qy2")
MUDNN_VERSION=rc2.7.0
MCCL_VERSION=rc1.7.0
TORCH_MUSA_VERSION=1.3.0
WHL_URL="oss.mthreads.com/ai-product/release-rc/torch_musa/rc${TORCH_MUSA_VERSION}"
SW_TAG=rc3.1.0

for arch in "${archs[@]}"; do
for version in "${versions[@]}"; do
    MUSA_TOOLKITS_URL="https://oss.mthreads.com/release-rc/cuda_compatible/${SW_TAG}/musa_toolkits_${SW_TAG}.tar.gz"
    MUDNN_URL="https://oss.mthreads.com/release-rc/cuda_compatible/${SW_TAG}/mudnn_${MUDNN_VERSION}.tar.gz"
    MCCL_URL="https://oss.mthreads.com/release-rc/cuda_compatible/${SW_TAG}/mccl_${MCCL_VERSION}.tar.gz"

    TAG="${SW_TAG}-v${TORCH_MUSA_VERSION}-${arch}"
    command="bash build.sh -n sh-harbor.mthreads.com/mt-ai/musa-pytorch-release-py$version                  \
              -b ubuntu:20.04                                                                               \
              -f ./ubuntu/dockerfile.release                                                                \
              -v ${version:0:1}.${version:1}                                                                \
              -m ${MUSA_TOOLKITS_URL}                                                                       \
              -t ${TAG}                                                                                     \
              --mudnn_url ${MUDNN_URL}                                                                      \
              --mccl_url ${MCCL_URL}                                                                        \
              --torch_whl_url ${WHL_URL}/torch-2.0.0-cp$version-cp$version-linux_x86_64.whl           \
              --torch_musa_whl_url ${WHL_URL}/${arch}/torch_musa-${TORCH_MUSA_VERSION}-cp$version-cp$version-linux_x86_64.whl \
              --torchaudio_whl_url ${WHL_URL}/2.2.2+cefdb36-cp$version-cp$version-linux_x86_64.whl \
              --torchvision_whl_url ${WHL_URL}/0.17.2+c1d70fe-cp$version-cp$version-linux_x86_64.whl \
              --release"
    $command
    if [ $? -ne 0 ]; then
      echo -e "\033[31mFAILED TO BUILD! COMMAND:${command}  \033[0m"
      exit 1
    fi
done
done
