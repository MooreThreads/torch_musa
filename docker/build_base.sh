#!/bin/bash
# Use this script to build the base docker image(base dev docker) for torch_musa, which includes the
# pytorch source code and conda environment
# example: DOCKER_BUILD_DIR=/tmp/torch_musa_base_docker_build \
#          PYTORCH_REPO_ROOT_PATH=~/tmp \
#          bash docker/build_base.sh -i pytorch2.0.0 -s ubuntu:20.04
set -e

# Please check https://github.com/pytorch/vision and https://pytorch.org/audio/main/installation.html for the version compatibility
PYTORCH_TAG="v2.0.0"
VISION_TAG="v0.15.2"
AUDIO_TAG="v2.0.1"

IMAGE_DOCKER_NAME=NULL
TAG=latest
PYTHON_VERSION="3.8"
OS_NAME=ubuntu:20.04

usage() {
  echo -e "\033[1;32mThis script is used to build base docker image which contains pytorch source code. \033[0m"
  echo -e "\033[1;32mParameters usage: \033[0m"
  echo -e "\033[32m    -n/--name                  : Name of the docker image. \033[0m"
  echo -e "\033[32m    -t/--tag                   : Tag of the docker image. \033[0m"
  echo -e "\033[32m    -s/--sys                   : The operating system, for example ubuntu:20.04 \033[0m"
  echo -e "\033[32m    -v/--python_version        : The python version used by torch_musa. \033[0m"
  echo -e "\033[32m    -h/--help                  : Help information. \033[0m"
}

# parse parameters
parameters=$(getopt -o n:t:s:v:h:: --long name:,tag:,sys:,python_verison:,help::, -n "$0" -- "$@")
[ $? -ne 0 ] && { echo -e "\033[34mTry '$0 --help' for more information. \033[0m"; exit 1; }

eval set -- "$parameters"

while true;do
  case "$1" in
    -n|--name) IMAGE_DOCKER_NAME=$2; shift 2;;
    -t|--tag) TAG=$2; shift 2;;
    -s|--sys) OS_NAME=$2; shift 2;;
    -v|--python_verison) PYTHON_VERSION=$2; shift 2;;
    -h|--help) usage; exit ;;
    --) shift ; break ;;
    *) usage; exit 1 ;;
  esac
done

# parse dockerfile and os
read OS VERSION_NUMBER <<< `echo $OS_NAME | awk -F: '{print $1, $2}'`
OS=$(echo "$OS" | tr '[:upper:]' '[:lower:]')
DOCKER_FILE="${OS}/dockerfile.base"
if [ ${OS} == "ubuntu" ]
then
  UBUNTU_VERSION=$VERSION_NUMBER
elif [ ${OS} == "centos" ]
then
  CENTOS_VERSION=$VERSION_NUMBER
else
  echo -e "\033[34mUnsupported operating system. \033[0m"
  exit 1
fi

PYTORCH_ROOT_DIR=${PYTORCH_REPO_ROOT_PATH:-${HOME}}

if [ ! -d "$PYTORCH_ROOT_DIR/pytorch" ]; then
  # if pytorch repo not exists, download it firstly
  echo "pytorch will be downloaded to ${PYTORCH_ROOT_DIR}"
  git clone -b ${PYTORCH_TAG} https://github.com/pytorch/pytorch.git --depth=1 $PYTORCH_ROOT_DIR/pytorch
  git submodule update --init --recursive
fi
if [ ! -d "$PYTORCH_ROOT_DIR/vision" ]; then
  # if torchvision repo not exists, download it firstly
  echo "torchvision will be downloaded to ${cd}"
  git clone -b ${VISION_TAG} https://github.com/pytorch/vision.git --depth=1 $PYTORCH_ROOT_DIR/vision
  git submodule update --init --recursive
fi
if [ ! -d "$PYTORCH_ROOT_DIR/audio" ]; then
  # if torchaudio repo not exists, download it firstly
  echo "torchaudio will be downloaded to ${PYTORCH_ROOT_DIR}"
  git clone -b ${AUDIO_TAG} https://github.com/pytorch/audio.git --depth=1 $PYTORCH_ROOT_DIR/audio
  git submodule update --init --recursive
fi

# BUILD_DIR is docker build context
BUILD_DIR=${DOCKER_BUILD_DIR:-$(pwd)/tmp}
sudo mkdir -p $BUILD_DIR
CUR_ROOT=$(cd "$(dirname "$0")"; pwd)
DOCKER_FILE=$CUR_ROOT/$DOCKER_FILE
sudo cp -r $CUR_ROOT/common $BUILD_DIR/
sudo cp -r $PYTORCH_ROOT_DIR/pytorch $BUILD_DIR
sudo cp -r $PYTORCH_ROOT_DIR/vision $BUILD_DIR
sudo cp -r $PYTORCH_ROOT_DIR/audio $BUILD_DIR

pushd $BUILD_DIR
ENV_NAME="py"$(echo ${PYTHON_VERSION} | tr -d '.')
# build docker image
build_docker_cmd_prefix="docker build --no-cache --network=host "
build_docker_cmd=${build_docker_cmd_prefix}"--build-arg UBUNTU_VERSION=${UBUNTU_VERSION}         \
                                            --build-arg CENTOS_VERSION=${CENTOS_VERSION}         \
                                            --build-arg ENV_NAME=${ENV_NAME}                     \
                                            --build-arg PYTHON_VERSION=${PYTHON_VERSION}         \
                                            -t ${IMAGE_DOCKER_NAME}:${TAG}                       \
                                            -f ${DOCKER_FILE} ."
echo -e "\033[34mbuild_docker cmd: "$build_docker_cmd"\033[0m"
eval $build_docker_cmd
popd

sudo rm -rf $BUILD_DIR
