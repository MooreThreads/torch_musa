#!/bin/bash  
versions=("38")  
archs=("qy2")
MUDNN_VERSION=dev2.6.0
WHL_URL="oss.mthreads.com/ai-product/release-rc/torch_musa/rc1.2.0" 

for arch in "${archs[@]}"; do
for version in "${versions[@]}"; do
    MUSA_TOOLKITS_URL="https://oss.mthreads.com/release-rc/cuda_compatible/dev3.0.0/${arch}/musa_toolkits_dev3.0.0-${arch}.tar.gz"
    MUDNN_URL="https://oss.mthreads.com/release-rc/cuda_compatible/dev3.0.0/${arch}/mudnn_${MUDNN_VERSION}-${arch}.tar.gz"
    
    if [ $arch=="qy1" ]; then
      echo "qy1 mccl..."
      MCCL_URL="https://oss.mthreads.com/release-rc/cuda_compatible/rc2.0.0/${arch}/mccl_rc1.4.0-${arch}.tar.gz"
    else
      echo "qy2 mccl..."
      MCCL_URL="https://oss.mthreads.com/release-rc/cuda_compatible/dev3.0.0/${arch}/mccl_dev1.6.0-${arch}.tar.gz"
    fi
    TAG="dev3.0.0-v1.2.0-${arch}"
    command="bash build.sh -n sh-harbor.mthreads.com/mt-ai/musa-pytorch-release-py$version                  \
              -b ubuntu:20.04                                                                               \
              -f ./ubuntu/dockerfile.release                                                                \
              -v ${version:0:1}.${version:1}                                                                \
              -m ${MUSA_TOOLKITS_URL}                                                                       \
              -t ${TAG}                                                                                     \
              --mudnn_url ${MUDNN_URL}                                                                      \
              --mccl_url ${MCCL_URL}                                                                        \
              --torch_whl_url ${WHL_URL}/torch-2.0.0-cp$version-cp$version-linux_x86_64.whl           \
              --torch_musa_whl_url ${WHL_URL}/${arch}/torch_musa-1.2.0-cp$version-cp$version-linux_x86_64.whl \
              --torchaudio_whl_url ${WHL_URL}/torchaudio-2.0.1+3b40834-cp38-cp38-linux_x86_64.whl \
              --torchvision_whl_url ${WHL_URL}/torchvision-0.15.2a0+fa99a53-cp38-cp38-linux_x86_64.whl \
              --release"
    $command
    if [ $? -ne 0 ]; then
      echo -e "\033[31mFAILED TO BUILD! COMMAND:${command}  \033[0m"
      exit 1
    fi
done
done
