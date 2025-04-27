#!/bin/bash
TORCH_MUSA_TAG='v2.0.0'
versions=("310")

TORCH_MUSA_VERSION=2.0.0
KINETO_TAG=v1.2.3
ALG_TAG=musa-1.12.1
THRUST_TAG=musa-1.12.1

function prepare_build_context() {
  # preprare files will be used when building docker image
  BUILD_DIR=${1:-$(pwd)/tmp}
  sudo mkdir -p $BUILD_DIR
  sudo git clone -b ${TORCH_MUSA_TAG} https://sh-code.mthreads.com/ai/torch_musa.git $BUILD_DIR/torch_musa
  sudo git clone -b ${KINETO_TAG} https://github.com/MooreThreads/kineto.git --depth 1 --recursive $BUILD_DIR/kineto
  sudo git clone -b ${ALG_TAG} https://github.com/MooreThreads/muAlg --depth 1 $BUILD_DIR/muAlg
  sudo git clone -b ${THRUST_TAG} https://github.com/MooreThreads/muThrust --depth 1 $BUILD_DIR/muThrust
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
  if [ "$ARCH" = "qy1" ]; then
    OSS_PREFIX=https://oss.mthreads.com/release-rc/cuda_compatible
    SW_TAG=rc4.0.1
    CC=Intel+Ubuntu
    MUDNN_VERSION=rc2.8.1
    MUSA_TOOLKITS_URL="${OSS_PREFIX}/${SW_TAG}/${CC}/musa_toolkits_${SW_TAG}.tar.gz"
    MUDNN_URL="${OSS_PREFIX}/${SW_TAG}/${CC}/mudnn_${MUDNN_VERSION}.tar.gz"
  elif [ "$ARCH" = "qy2" ]; then
    SW_TAG=rc4.0.0
    OSS_PREFIX=https://oss.mthreads.com/release-ci/computeQA/cuda_compatible/CI/release_musa_4.0.0/2025-04-13
    MUDNN_VERSION=rc2.8.0
    MCCL_VERSION=rc1.8.0
    MUSA_TOOLKITS_URL="${OSS_PREFIX}/musa_toolkits_install_full.tar.gz"
    MUDNN_URL="${OSS_PREFIX}/mudnn_${MUDNN_VERSION}.tar.gz"
    MCCL_URL="${OSS_PREFIX}/mccl_${MCCL_VERSION}.tar.gz"
  else
    SW_TAG=rc4.0.0
    OSS_PREFIX=https://oss.mthreads.com/release-ci/computeQA/cuda_compatible/CI/release_KUAE_2.0_for_PH1_M3D/2025-04-13
    MUDNN_VERSION=dev2.8.0
    MCCL_VERSION=dev1.8.0
    MUSA_TOOLKITS_URL="${OSS_PREFIX}/musa_toolkits_install_full.tar.gz"
    MUDNN_URL="${OSS_PREFIX}/mudnn_${MUDNN_VERSION}.PH1.tar.gz"
    MCCL_URL="${OSS_PREFIX}/mccl_${MCCL_VERSION}.PH1.tar.gz"
  fi
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
