#!/bin/bash
TORCH_MUSA_TAG='main'
versions=("38")


BUILD_GPU_ARCH=${BUILD_GPU_ARCH:-qy2}

if [ ${BUILD_GPU_ARCH} = "qy2" ]; then
  MUSA_TOOLKITS_URL="https://oss.mthreads.com/release-rc/cuda_compatible/dev3.0.0/qy2/musa_toolkits_dev3.0.0-qy2.tar.gz"
  MUDNN_URL="https://oss.mthreads.com/release-rc/cuda_compatible/dev3.0.0/qy2/mudnn_dev2.6.0-qy2.tar.gz"
  MCCL_URL="https://oss.mthreads.com/release-rc/cuda_compatible/dev3.0.0/qy2/mccl_dev1.6.0-qy2.tar.gz"
  TAG="dev3.0.0-v1.1.0-qy2"
elif [ ${BUILD_GPU_ARCH} = "qy1" ]; then
  MUSA_TOOLKITS_URL="https://oss.mthreads.com/release-rc/cuda_compatible/dev3.0.0/qy1/musa_toolkits_dev3.0.0-qy1.tar.gz"
  MUDNN_URL="https://oss.mthreads.com/release-rc/cuda_compatible/dev3.0.0/qy1/mudnn_dev2.6.0-qy1.tar.gz"
  MCCL_URL="https://oss.mthreads.com/release-rc/cuda_compatible/rc2.0.0/qy1/mccl_rc1.4.0-qy1.tar.gz"
  TAG="dev3.0.0-v1.1.0-qy1"
fi

function prepare_build_context() {
  # preprare files will be used when building docker image
  BUILD_DIR=${1:-$(pwd)/tmp}
  sudo mkdir -p $BUILD_DIR
  sudo git clone -b ${TORCH_MUSA_TAG} https://sh-code.mthreads.com/ai/torch_musa.git $BUILD_DIR/torch_musa
  CUR_ROOT=$(cd "$(dirname "$0")"; pwd)
  sudo cp -r $CUR_ROOT/common $BUILD_DIR/
  integration_data_path=/jfs/torch_musa_integration/data.tar.gz
  if [ ! -f ${integration_data_path} ]; then
    echo -e "\033[31mDirectory '${integration_data_path}' in /jfs should be mounted first! \033[0m"
    exit 1
  fi
  echo "Copying data..."
  sudo cp ${integration_data_path} ${BUILD_DIR}/
  pushd ${BUILD_DIR}/
  sudo tar xzvf data.tar.gz
  popd
  echo "Copying data finished."
}

NO_PREPARE=${NO_PREPARE:-0}
if [ ${NO_PREPARE} = "0" ]; then
  echo "Preparing data..."
  prepare_build_context
  echo "Preparing data finished."
fi


for version in "${versions[@]}"; do  
    command="bash build.sh -n sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev-py$version    \
        -b sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev-py$version:base-pytorch2.0.0     \
        -f ./ubuntu/dockerfile.dev                                                        \ 
        -t ${TAG}                                                                         \       
        -m ${MUSA_TOOLKITS_URL}                                                           \
        --mudnn_url ${MUDNN_URL}                                                          \
        --mccl_url  ${MCCL_URL}                                                           \    
        --torch_musa_tag  ${TORCH_MUSA_TAG}                                               \    
        --no_prepare"
    $command
    if [ $? -ne 0 ]; then
      echo -e "\033[31mFAILED TO BUILD! COMMAND:${command}  \033[0m"
      exit 1
    fi
done


sudo rm -rf $BUILD_DIR
