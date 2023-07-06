#!/bin/bash
set -e

PYTORCH_TAG=v2.0.0
IMAGE_DOCKER_NAME=NULL
TAG=latest
PYTHON_VERSION="3.8"
RELEASE=0
OS_NAME=NULL
MUSA_TOOLKITS_URL=""
MUDNN_URL=""
TORCH_WHL_URL=""
TORCH_MUSA_WHL_URL=""

usage() {
  echo -e "\033[1;32mThis script is used to build docker image for torch_musa. \033[0m"
  echo -e "\033[1;32mParameters usage: \033[0m"
  echo -e "\033[32m    -i/--image_docker_name     : Name of the docker image. \033[0m"
  echo -e "\033[32m    -t/--tag                   : Tag of the docker image. \033[0m"
  echo -e "\033[32m    -s/--sys                   : The operating system, for example ubuntu:20.04 \033[0m"
  echo -e "\033[32m    -v/--python_version        : The python version used by torch_musa. \033[0m"
  echo -e "\033[32m    -m/--musa_toolkits_url     : The download link of MUSA ToolKit. \033[0m"
  echo -e "\033[32m    -n/--mudnn_url             : The download link of MUDNN. \033[0m"
  echo -e "\033[32m    -r/--release               : The build pattern, default develop. \033[0m"
  echo -e "\033[32m    --torch_whl_url)           : The download link of torch wheel. \033[0m"
  echo -e "\033[32m    --torch_musa_whl_url)      : The download link of torch_musa wheel. \033[0m"
  echo -e "\033[32m    -h/--help                  : Help information. \033[0m"
}

# parse parameters
parameters=$(getopt -o ri:t:s:v:m:n:h:: --long image_docker_name:,tag:,sys:,python_verison:,musa_toolkits_url:,mudnn_url:,release,torch_whl_url:,torch_musa_whl_url:,help::, -n "$0" -- "$@")
[ $? -ne 0 ] && { echo -e "\033[34mTry '$0 --help' for more information. \033[0m"; exit 1; }

eval set -- "$parameters"

while true;do
  case "$1" in
    -i|--image_docker_name) IMAGE_DOCKER_NAME=$2; shift 2;;
    -t|--tag) TAG=$2; shift 2;;
    -s|--sys) OS_NAME=$2; shift 2;;
    -v|--python_verison) PYTHON_VERSION=$2; shift 2;;
    -m|--musa_toolkits_url) MUSA_TOOLKITS_URL=$2; shift 2;;
    -n|--mudnn_url) MUDNN_URL=$2; shift 2;;
    -r|--release) RELEASE=1; shift ;;
    --torch_whl_url) TORCH_WHL_URL=$2; shift 2;;
    --torch_musa_whl_url) TORCH_MUSA_WHL_URL=$2; shift 2;;
    -h|--help) usage; exit ;;
    --) shift ; break ;;
    *) usage; exit 1 ;;
  esac
done


# parse dockerfile and os
read OS VERSION_NUMBER <<< `echo $OS_NAME | awk -F: '{print $1, $2}'`
OS=$(echo "$OS" | tr '[:upper:]' '[:lower:]')
DOCKER_FILE="${OS}/dockerfile"
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

if [ ${RELEASE} -eq 1 ]
then
  DOCKER_FILE=$DOCKER_FILE".release"
else
  DOCKER_FILE=$DOCKER_FILE".dev"
fi


PYTORCH_ROOT_DIR=${PYTORCH_REPO_ROOT_PATH:-${HOME}}

if [ ! -d "$PYTORCH_ROOT_DIR/pytorch" ]; then
  # if pytorch repo not exists, download it firstly
  echo "pytorch will be downloaded to ${PYTORCH_ROOT_DIR}"
  git clone -b ${PYTORCH_TAG} https://github.com/pytorch/pytorch.git --depth=1 $PYTORCH_ROOT_DIR/pytorch
  git submodule update --init --recursive
fi

BUILD_DIR=$(pwd)/tmp
mkdir -p $BUILD_DIR
CUR_ROOT=$(cd "$(dirname "$0")"; pwd)
cp -r $CUR_ROOT/common $BUILD_DIR/
cp -r $CUR_ROOT/ubuntu $BUILD_DIR
cp $CUR_ROOT/../requirements.txt $BUILD_DIR


pushd $BUILD_DIR
if [ ${RELEASE} -eq 0 ]; then
  # prepare pytorch and torch_musa
  cp -r $PYTORCH_ROOT_DIR/pytorch $BUILD_DIR
  git clone https://github.mthreads.com/mthreads/torch_musa.git
fi

# build docker image
build_docker_cmd_prefix="docker build --no-cache --network=host "
build_docker_cmd=${build_docker_cmd_prefix}"--build-arg UBUNTU_VERSION=${UBUNTU_VERSION}                     \
                                            --build-arg CENTOS_VERSION=${CENTOS_VERSION}                     \
                                            --build-arg PYTHON_VERSION=${PYTHON_VERSION}                     \
                                            --build-arg MUSA_TOOLKITS_URL=${MUSA_TOOLKITS_URL}               \
                                            --build-arg MUDNN_URL=${MUDNN_URL}                               \
                                            --build-arg TORCH_WHL_URL=${TORCH_WHL_URL}                       \
                                            --build-arg TORCH_MUSA_WHL_URL=${TORCH_MUSA_WHL_URL}             \
                                            -t ${IMAGE_DOCKER_NAME}:${TAG}                                   \
                                            -f ${DOCKER_FILE} ."
echo -e "\033[34mbuild_docker cmd: "$build_docker_cmd"\033[0m"
eval $build_docker_cmd
popd
