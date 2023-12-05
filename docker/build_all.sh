#!/bin/bash  
VISION_TAG="v0.16.0"
TORCH_MUSA_TAG='dev1.5.1'
versions=("38" "39" "310")  

MUSA_TOOLKITS_URL="https://oss.mthreads.com/release-rc/cuda_compatible/dev1.5.1/qy1/musa_toolkits_dev1.5.1-qy1.tar.gz"
MUDNN_URL="https://oss.mthreads.com/release-rc/cuda_compatible/dev1.5.1/qy1/mudnn_rtm2.3.0-qy1.tar"
MCCL_URL="https://oss.mthreads.com/release-rc/cuda_compatible/dev1.5.1/qy1/mccl_dev1.3.0.tar "
TAG="dev1.5.1-qy1"
WHL_URL="oss.mthreads.com/ai-product/daily/framework/torch_musa/20231118_torch_musa" 

function prepare_build_context() {
  # preprare files will be used when building docker image
  BUILD_DIR=${1:-$(pwd)/tmp}
  sudo mkdir -p $BUILD_DIR
  sudo git clone -b ${TORCH_MUSA_TAG} https://github.mthreads.com/mthreads/torch_musa.git $BUILD_DIR/torch_musa
  CUR_ROOT=$(cd "$(dirname "$0")"; pwd)
  sudo cp -r $CUR_ROOT/common $BUILD_DIR/
}

prepare_build_context



for version in "${versions[@]}"; do  
    command="bash build.sh -n sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev-py$version    \
        -b sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev-py$version:base-pytorch2.0.0     \
        -f ./ubuntu/dockerfile.dev                                                        \ 
        -t ${TAG}                                                                         \       
        -m ${MUSA_TOOLKITS_URL}                                                           \
        --mudnn_url ${MUDNN_URL}                                                          \
        --mccl_url  ${MCCL_URL}                                                           \    
        --no_prepare"
    $command
    if [ $? -ne 0 ]; then
      echo -e "\033[31mFAILED TO BUILD! COMMAND:${command}  \033[0m"
      exit 1
    fi
done


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
              --torch_musa_whl_url ${WHL_URL}/torch_musa-2.0.0-cp$version-cp$version-linux_x86_64.whl \
              --release"
    $command
    if [ $? -ne 0 ]; then
      echo -e "\033[31mFAILED TO BUILD! COMMAND:${command}  \033[0m"
      exit 1
    fi
done

sudo rm -rf $BUILD_DIR