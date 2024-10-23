#!/bin/bash
TORCH_MUSA_TAG='v1.3.0'
versions=("38" "39" "310")
archs=("qy1" "qy2")

SW_TAG=rc3.1.0
MUDNN_VERSION=rc2.7.0
MCCL_VERSION=rc1.7.0
TORCH_MUSA_VERSION=1.3.0
MUSA_TOOLKITS_URL="https://oss.mthreads.com/release-rc/cuda_compatible/${SW_TAG}/musa_toolkits_${SW_TAG}.tar.gz"
MUDNN_URL="https://oss.mthreads.com/release-rc/cuda_compatible/${SW_TAG}/mudnn_${MUDNN_VERSION}.tar.gz"
MCCL_URL="https://oss.mthreads.com/release-rc/cuda_compatible/${SW_TAG}/mccl_${MCCL_VERSION}.tar.gz"

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


for arch in "${archs[@]}"; do
for version in "${versions[@]}"; do
    TAG="${SW_TAG}-v${TORCH_MUSA_VERSION}-${arch}"
    command="bash build.sh -n sh-harbor.mthreads.com/mt-ai/musa-pytorch2.2.0-dev-py$version   \
        -b sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev-py$version:base-$TAG                 \
        -f ./ubuntu/dockerfile.dev                                                            \
        -v ${version:0:1}.${version:1}                                                        \ 
        -t ${TAG}                                                                             \       
        -m ${MUSA_TOOLKITS_URL}                                                               \
        --mudnn_url ${MUDNN_URL}                                                              \
        --mccl_url  ${MCCL_URL}                                                               \    
        --torch_musa_tag  ${TORCH_MUSA_TAG}                                                   \    
        --no_prepare"
    $command
    if [ $? -ne 0 ]; then
      echo -e "\033[31mFAILED TO BUILD! COMMAND:${command}  \033[0m"
      exit 1
    fi
done


sudo rm -rf $BUILD_DIR
