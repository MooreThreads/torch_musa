#!/bin/bash
# Use this script to build docker image for torch_musa
# example: DOCKER_BUILD_DIR=/data/torch_musa_docker_build \
#          TORCH_VISION_REPO_ROOT_PATH=/tmp \
#          bash docker/build.sh -i torch_musa_dev \
#                               -b sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev:base-pytorch-v2.0.0 \
#                               -f docker/ubuntu/dockerfile.dev \
#                               -m https://oss.mthreads.com/release-rc/cuda_compatible/dev1.5.0/QY2/musa_toolkits_dev1.5.0-qy2.tar.gz \
#                               -n https://oss.mthreads.com/release-rc/cuda_compatible/dev1.5.0/QY2/mudnn_dev2.3.0-qy2.tar
#                               --mccl_url https://oss.mthreads.com/release-rc/cuda_compatible/dev1.5.1/qy1/mccl_dev1.3.0.tar
set -e

IMAGE_DOCKER_NAME=NULL
TAG=latest
PYTHON_VERSION="3.8"
RELEASE=0
BASE_IMG=NULL
DOCKER_FILE=""
MUSA_TOOLKITS_URL=""
MUDNN_URL=""
MCCL_URL=""
TORCH_WHL_URL=""
TORCH_MUSA_WHL_URL=""
VISION_TAG="v0.16.0"

usage() {
  echo -e "\033[1;32mThis script is used to build docker image for torch_musa. \033[0m"
  echo -e "\033[1;32mParameters usage: \033[0m"
  echo -e "\033[32m    -n/--name                  : Name of the docker image. \033[0m"
  echo -e "\033[32m    -t/--tag                   : Tag of the docker image. \033[0m"
  echo -e "\033[32m    -b/--base_img              : The base docker image, for example ubuntu:20.04. \033[0m"
  echo -e "\033[32m    -f/--docker_file           : The path of docker file. \033[0m"
  echo -e "\033[32m    -v/--python_version        : The python version used by torch_musa. \033[0m"
  echo -e "\033[32m    -m/--musa_toolkits_url     : The download link of MUSA ToolKit. \033[0m"
  echo -e "\033[32m    --mudnn_url)               : The download link of MUDNN. \033[0m"
  echo -e "\033[32m    -r/--release               : The build pattern, default develop. \033[0m"
  echo -e "\033[32m    --mccl_url)                : The download link of MCCL. \033[0m"
  echo -e "\033[32m    --torch_whl_url)           : The download link of torch wheel. \033[0m"
  echo -e "\033[32m    --torch_musa_whl_url)      : The download link of torch_musa wheel. \033[0m"
  echo -e "\033[32m    -h/--help                  : Help information. \033[0m"
}

# parse parameters
parameters=$(getopt -o rn:t:b:f:v:m:h:: --long name:,tag:,base_img:,docker_file:,python_verison:,musa_toolkits_url:,mudnn_url:,release,mccl_url:,torch_whl_url:,torch_musa_whl_url:,help::, -n "$0" -- "$@")
[ $? -ne 0 ] && { echo -e "\033[34mTry '$0 --help' for more information. \033[0m"; exit 1; }

eval set -- "$parameters"

while true;do
  case "$1" in
    -n|--name) IMAGE_DOCKER_NAME=$2; shift 2;;
    -t|--tag) TAG=$2; shift 2;;
    -b|--base_img) BASE_IMG=$2; shift 2;;
    -f|--docker_file) DOCKER_FILE=$2; shift 2;;
    -v|--python_verison) PYTHON_VERSION=$2; shift 2;;
    -m|--musa_toolkits_url) MUSA_TOOLKITS_URL=$2; shift 2;;
    --mudnn_url) MUDNN_URL=$2; shift 2;;
    -r|--release) RELEASE=1; shift ;;
    --mccl_url) MCCL_URL=$2; shift 2;;
    --torch_whl_url) TORCH_WHL_URL=$2; shift 2;;
    --torch_musa_whl_url) TORCH_MUSA_WHL_URL=$2; shift 2;;
    -h|--help) usage; exit ;;
    --) shift ; break ;;
    *) usage; exit 1 ;;
  esac
done

function prepare_build_context() {
  # preprare files will be used when building docker image
  BUILD_DIR=${1:-$(pwd)/tmp}
  if [ ${RELEASE} -eq 0 ]; then
    # add projects here which needs to be installed after torch_musa installation done
    TORCH_VISION_ROOT_DIR=${TORCH_VISION_REPO_ROOT_PATH:-${HOME}}
    if [ ! -d "$TORCH_VISION_ROOT_DIR/vision" ]; then
      echo "torchvision will be downloaded to ${TORCH_VISION_ROOT_DIR}"
      git clone -b ${VISION_TAG}  https://github.com/pytorch/vision.git --depth=1 $TORCH_VISION_ROOT_DIR/vision
    fi
    sudo cp -r $TORCH_VISION_ROOT_DIR/vision $BUILD_DIR
    sudo git clone https://github.mthreads.com/mthreads/torch_musa.git $BUILD_DIR/torch_musa
  fi
  CUR_ROOT=$(cd "$(dirname "$0")"; pwd)
  sudo cp -r $CUR_ROOT/common $BUILD_DIR/
}

function build_dev_base_docker_image() {
  echo "Please run bash build_base.sh "
  exit 0
}

function build_docker_image() {
  BUILD_CONTEXT_DIR=$1
  build_docker_cmd_prefix="docker build --no-cache --network=host "
  build_docker_cmd=${build_docker_cmd_prefix}"--build-arg BASE_IMG=${BASE_IMG}
                                              --build-arg PYTHON_VERSION=${PYTHON_VERSION}                     \
                                              --build-arg MUSA_TOOLKITS_URL=${MUSA_TOOLKITS_URL}               \
                                              --build-arg MUDNN_URL=${MUDNN_URL}                               \
                                              --build-arg TORCH_WHL_URL=${TORCH_WHL_URL}                       \
                                              --build-arg TORCH_MUSA_WHL_URL=${TORCH_MUSA_WHL_URL}             \
                                              --build-arg MCCL_URL=${MCCL_URL}                                 \
                                              -t ${IMAGE_DOCKER_NAME}:${TAG}                                   \
                                              -f ${DOCKER_FILE} ${BUILD_CONTEXT_DIR}"
  echo -e "\033[34mbuild_docker cmd: "$build_docker_cmd"\033[0m"
  eval $build_docker_cmd
}

BUILD_DIR=${DOCKER_BUILD_DIR:-$(pwd)/tmp}
sudo mkdir -p $BUILD_DIR
prepare_build_context ${BUILD_DIR}
build_docker_image $BUILD_DIR
sudo rm -rf $BUILD_DIR

