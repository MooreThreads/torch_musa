#!/bin/bash
set -e

IMAGE_DOCKER_NAME=NULL
TAG=latest
PYTHON_VERSION="3.8"
RELEASE=0
OS_NAME=NULL
MUSA_TOOLKITS_URL="https://oss.mthreads.com/release-ci/computeQA/musa/history/20230425musa_toolkite1841ca3fcompute_musa_pkg1427/musa_toolkits_install_full.tar.gz"
MUDNN_URL="http://oss.mthreads.com/release-ci/computeQA/ai/newest/mudnn.tar"
GIT_CREDENTIAL_FILE=""
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
  echo -e "\033[32m    -c/--git_credential_file)  : The credential of git. \033[0m"
  echo -e "\033[32m    --torch_whl_url)           : The download link of torch wheel. \033[0m"
  echo -e "\033[32m    --torch_musa_whl_url)      : The download link of torch_musa wheel. \033[0m"
  echo -e "\033[32m    -h/--help                  : Help information. \033[0m"
}

# parse parameters
parameters=$(getopt -o ri:t:s:v:m:n:c:h:: --long image_docker_name:,tag:,sys:,python_verison:,musa_toolkits_url:,mudnn_url:,release,git_credential_file:,torch_whl_url:,torch_musa_whl_url:,help::, -n "$0" -- "$@")
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
    -c|--git_credential_file) GIT_CREDENTIAL_FILE=$2; shift 2;;
    --torch_whl_url) TORCH_WHL_URL=$2; shift 2;;
    --torch_musa_whl_url) TORCH_MUSA_WHL_URL=$2; shift 2;;
    -h|--help) usage; exit ;;
    --) shift ; break ;;
    *) usage; exit 1 ;;
  esac
done

if [ ${RELEASE} -eq 0 ] && [ -z "$GIT_CREDENTIAL_FILE" ]; then
  echo -e "\033[34mgit credential is needed, which include your name and password of git, cause torch_musa repository will be cloned through https.  \033[0m"
  exit 1
fi

# parsing dockerfile & os
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

# build docker image
if [ ${RELEASE} -eq 0 ]
then
  build_docker_cmd_prefix="DOCKER_BUILDKIT=1 sudo docker build  --no-cache --network=host \
                           --secret id=gitCredential,src=${GIT_CREDENTIAL_FILE} "
else
  build_docker_cmd_prefix="sudo docker build --no-cache --network=host "
fi
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
