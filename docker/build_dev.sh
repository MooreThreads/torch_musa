#!/bin/bash
TORCH_MUSA_TAG='v1.3.2'
versions=("310")

SW_TAG=dev4.0.0
MUDNN_VERSION=dev2.8.0
MCCL_VERSION=dev1.8.0
TORCH_MUSA_VERSION=1.3.2
OSS_PREFIX=https://oss.mthreads.com/release-rc/cuda_compatible

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

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')
ARCH="ph1"
if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ] || [ "$GPU" = "MTTS80ES" ]; then
  ARCH="qy1"
elif [ "$GPU" = "MTTS4000" ]; then
  ARCH="qy2"
fi

for version in "${versions[@]}"; do
  if [ "$ARCH" = "qy1" ] || [ "$ARCH" = "qy2" ]; then
    CC=cc2.2
  else
    CC=cc3.1
  fi
  MUSA_TOOLKITS_URL="${OSS_PREFIX}/${SW_TAG}/${CC}/musa_toolkits_${SW_TAG}.tar.gz"
  MUDNN_URL="${OSS_PREFIX}/${SW_TAG}/${CC}/mudnn_${MUDNN_VERSION}.${CC}.tar.gz"
  MCCL_URL="${OSS_PREFIX}/${SW_TAG}/${CC}/mccl_${MCCL_VERSION}.${CC}.tar.gz"
  TAG="${SW_TAG}-v${TORCH_MUSA_VERSION}-${ARCH}"
  command="bash build.sh -n sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev-py$version           \
      -b sh-harbor.mthreads.com/mt-ai/musa-pytorch-base-py$version:${SW_TAG}-${TORCH_MUSA_TAG} \
      -f ./ubuntu/dockerfile.dev                                                               \
      -v ${version:0:1}.${version:1}                                                           \
      -t ${TAG}                                                                                \
      -m ${MUSA_TOOLKITS_URL}                                                                  \
      --mudnn_url ${MUDNN_URL}                                                                 \
      --mccl_url  ${MCCL_URL}                                                                  \
      --torch_musa_tag  ${TORCH_MUSA_TAG}                                                      \
      --no_prepare"
  $command
  if [ $? -ne 0 ]; then
    echo -e "\033[31mFAILED TO BUILD! COMMAND:${command}  \033[0m"
    exit 1
  fi
done


sudo rm -rf $BUILD_DIR
