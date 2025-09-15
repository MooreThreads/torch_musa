#!/bin/bash
set -e

# Environment variables used when building torch_musa
#
#   TORCH_MUSA_ARCH_LIST
#     specify which MUSA architectures to build for.
#     ie 'TORCH_MUSA_ARCH_LIST="21;22"'
#

CUR_DIR=$(
  cd $(dirname $0)
  pwd
)
TORCH_MUSA_HOME=$CUR_DIR
PYTORCH_PATH=${PYTORCH_REPO_PATH:-$(realpath ${TORCH_MUSA_HOME}/../pytorch)}
TORCH_PATCHES_DIR=${TORCH_MUSA_HOME}/torch_patches/
KINETO_URL=${KINETO_URL:-https://github.com/MooreThreads/kineto.git}
KINETO_TAG=v2.0.1

BUILD_WHEEL=0
DEBUG_MODE=0
ASAN_MODE=0
BUILD_TORCH=1
BUILD_TORCH_MUSA=1
USE_KINETO=${USE_KINETO:-1}
ONLY_PATCH=0
CLEAN=0
COMPILE_FP64=1
PYTORCH_TAG=v2.5.0
PYTORCH_BUILD_VERSION="${PYTORCH_TAG:1}"
PYTORCH_BUILD_NUMBER=0 # This is used for official torch distribution.
USE_MKL=${USE_MKL:-1}
USE_STATIC_MKL=${USE_STATIC_MKL:-1}
USE_MCCL=${USE_MCCL:-1}
MUSA_DIR="/usr/local/musa"
UPDATE_MUSA=0
UPDATE_DAILY_MUSA=0

usage() {
  echo -e "\033[1;32mThis script is used to build PyTorch and Torch_MUSA. \033[0m"
  echo -e "\033[1;32mParameters usage: \033[0m"
  echo -e "\033[32m    --all         : Means building both PyTorch and Torch_MUSA. \033[0m"
  echo -e "\033[32m    --fp64        : Means compiling fp64 data type in kernels using mcc in Torch_MUSA. \033[0m"
  echo -e "\033[32m    --update_musa : Update latest RELEASED MUSA software stack. \033[0m"
  echo -e "\033[32m    --update_daily_musa : Update latest DAILY MUSA software stack. \033[0m"
  echo -e "\033[32m    -m/--musa     : Means building Torch_MUSA only. \033[0m"
  echo -e "\033[32m    -t/--torch    : Means building original PyTorch only. \033[0m"
  echo -e "\033[32m    -d/--debug    : Means building in debug mode. \033[0m"
  echo -e "\033[32m    -a/--asan     : Means building in asan mode. \033[0m"
  echo -e "\033[32m    -c/--clean    : Means cleaning everything that has been built. \033[0m"
  echo -e "\033[32m    -p/--patch    : Means applying patches only. \033[0m"
  echo -e "\033[32m    -w/--wheel    : Means generating wheel after building. \033[0m"
  echo -e "\033[32m    -n/--no_kineto : Disable kineto. \033[0m"
  echo -e "\033[32m    -h/--help     : Help information. \033[0m"
}

# parse paremters
parameters=$(getopt -o +mtdacpwnh --long all,fp64,update_musa,update_daily_musa,musa,torch,debug,asan,clean,patch,wheel,no_kineto,help, -n "$0" -- "$@")
[ $? -ne 0 ] && {
  echo -e "\033[34mTry '$0 --help' for more information. \033[0m"
  exit 1
}

eval set -- "$parameters"

while true; do
  case "$1" in
  --all)
    BUILD_TORCH=1
    BUILD_TORCH_MUSA=1
    shift
    ;;
  --fp64)
    COMPILE_FP64=1
    shift
    ;;
  --update_musa)
    UPDATE_MUSA=1
    shift
    ;;
  --update_daily_musa)
    UPDATE_DAILY_MUSA=1
    shift
    ;;
  -m | --musa)
    BUILD_TORCH_MUSA=1
    BUILD_TORCH=0
    shift
    ;;
  -t | --torch)
    BUILD_TORCH_MUSA=0
    BUILD_TORCH=1
    shift
    ;;
  -d | --debug)
    DEBUG_MODE=1
    shift
    ;;
  -a | --asan)
    ASAN_MODE=1
    shift
    ;;
  -c | --clean)
    CLEAN=1
    shift
    ;;
  -w | --wheel)
    BUILD_WHEEL=1
    shift
    ;;
  -n | --no_kineto)
    USE_KINETO=0
    shift
    ;;
  -p | --patch)
    ONLY_PATCH=1
    shift
    ;;
  -h | --help)
    usage
    exit
    ;;
  --)
    shift
    break
    ;;
  *)
    usage
    exit 1
    ;;
  esac
done

cmd_check(){
  cmd="$1"
  if command -v ${cmd} >/dev/null 2>&1; then 
    echo "- cmd exist  : ${cmd}"
  else
    echo -e "\033[34m- cmd does not exist, automatically install \"${cmd}\"\033[0m"
    pip install -r ${TORCH_MUSA_HOME}/requirements.txt # extra requirements
  fi
}

precommit_install(){
  cmd_check "pre-commit"
  root_dir="$(dirname "$(realpath "${BASH_SOURCE:-$0}" )")"
  if [ ! -f ${root_dir}/.git/hooks/pre-commit ]; then
    pushd $root_dir
    pre-commit install 
    popd
  fi
}

precommit_install

clone_pytorch() {
  # if PyTorch repo exists already, we skip gitting clone PyTorch
  if [ -d ${PYTORCH_PATH} ]; then
    echo -e "\033[34mPyTorch repo path is ${PYTORCH_PATH} ...\033[0m"
    pushd ${PYTORCH_PATH}
    git checkout ${PYTORCH_TAG}
    echo -e "\033[34m Switch the Pytorch repo to tag ${PYTORCH_TAG} \033[0m"
    popd
  else
    ABSOLUTE_PATH=$(cd $(dirname ${PYTORCH_PATH}) && pwd)"/pytorch"
    echo -e "\033[34mUsing default pytorch repo path: ${ABSOLUTE_PATH}\033[0m"
    if [ ! -d "${PYTORCH_PATH}" ]; then
      pushd ${TORCH_MUSA_HOME}/..
      echo -e "\033[34mPyTorch repo does not exist, now git clone PyTorch to ${ABSOLUTE_PATH} ...\033[0m"
      git clone -b ${PYTORCH_TAG} https://github.com/pytorch/pytorch.git --depth=1
      popd
    fi
  fi
  # to make sure submodules are fetched
  pushd ${PYTORCH_PATH}
  update_submodule
}

apply_torch_patches() {
  # apply patches into PyTorch
  echo -e "\033[34mApplying patches to ${PYTORCH_PATH} ...\033[0m"
  # clean PyTorch before patching
  if [ -d "$PYTORCH_PATH/.git" ]; then
    echo -e "\033[34mStash and checkout the PyTorch environment before patching. \033[0m"
    pushd $PYTORCH_PATH
    git stash -u
    git checkout ${PYTORCH_TAG}
    popd
  fi

  for file in $(find ${TORCH_PATCHES_DIR} -type f -print); do
    if [ "${file##*.}"x = "patch"x ]; then
      echo -e "\033[34mapplying patch: $file \033[0m"
      pushd $PYTORCH_PATH
      git apply --check $file
      git apply $file
      popd
    fi
  done
}

update_kineto_source() {
  echo -e "\033[34mUpdating Kineto...\033[0m"
  pushd ${PYTORCH_PATH}
  rm -rf ${PYTORCH_PATH}/third_party/kineto
  git submodule update --init --recursive --depth 1
  rm -rf ${PYTORCH_PATH}/third_party/kineto
  popd
  echo -e "\033[34mUpdating KINETO_URL, might take a while...\033[0m"
  if [ -d /home/kineto ]; then
    pushd /home/kineto
    git checkout ${KINETO_TAG}
    git submodule update --init --recursive --depth 1
    popd
    cp -r /home/kineto ${PYTORCH_PATH}/third_party
  else
    git clone ${KINETO_URL} -b ${KINETO_TAG} --depth 1 --recursive ${PYTORCH_PATH}/third_party/kineto
  fi
}

# Since the initial environment uses musa kineto by default, we should
# manually redirect torch kineto's url && commitid if `USE_KINETO=0`.
# Currently, it's only required for the internal testing purpose.
revert_torch_kineto() {
  echo -e "\033[34mReverting to torch kineto...\033[0m"
  echo -e "\033[34mRemoving mupti...\033[0m"
  pushd ${PYTORCH_PATH}
  rm -rf third_party/kineto
  git submodule update --init --recursive third_party/kineto
  popd
}

update_submodule() {
  if [ -d ${PYTORCH_PATH}/third_party/kineto ]; then
    pushd ${PYTORCH_PATH}/third_party/kineto
    remote_url=$(git remote get-url origin)
    current_tag=$(git describe --tags --always)
    popd

    if [ ${USE_KINETO} -eq 0 ]; then
      if [ "${remote_url}" = "${KINETO_URL}" ]; then
        rm -rf ${PYTORCH_PATH}/third_party/kineto
      fi
      pushd ${PYTORCH_PATH}
      git submodule update --init --recursive --depth 1
      popd
    elif [ "${remote_url}" = "${KINETO_URL}" ] && [ "${current_tag}" = "${KINETO_TAG}" ]; then
      pushd ${PYTORCH_PATH}/third_party/kineto
      echo  -e "\033[34mUpdating KINETO submodule, might take a while...\033[0m"
      git submodule update --init --recursive
      popd
      if [ -d "/tmp/kineto" ]; then
        rm -rf /tmp/kineto
      fi
      mv ${PYTORCH_PATH}/third_party/kineto /tmp
      pushd ${PYTORCH_PATH}
      git submodule update --init --recursive --depth 1
      popd
      rm -rf ${PYTORCH_PATH}/third_party/kineto
      mv /tmp/kineto ${PYTORCH_PATH}/third_party
    else
      update_kineto_source
    fi
  elif [ ${USE_KINETO} -eq 1 ]; then
    update_kineto_source
  else
    pushd ${PYTORCH_PATH}
    git submodule update --init --recursive --depth 1
    popd
  fi
}

build_pytorch() {
  echo -e "\033[34mBuilding PyTorch...\033[0m"
  status=0
  if [ ! -d ${PYTORCH_PATH} ]; then
    echo -e "\033[34mAn error occurred while building PyTorch, the specified PyTorch repo [${PYTORCH_PATH}] does not exist \033[0m"
    exit 1
  fi

  pushd ${PYTORCH_PATH}
  pip install -r requirements.txt
  pip install -r ${TORCH_MUSA_HOME}/requirements.txt # extra requirements
  if [ $BUILD_WHEEL -eq 1 ]; then
    rm -rf dist
    pip uninstall torch -y
    PYTORCH_BUILD_NUMBER=${PYTORCH_BUILD_NUMBER} \
      PYTORCH_BUILD_VERSION=${PYTORCH_BUILD_VERSION} \
      DEBUG=${DEBUG_MODE} \
      USE_ASAN=${ASAN_MODE} \
      USE_STATIC_MKL=${USE_STATIC_MKL} \
      USE_MKL=${USE_MKL} \
      USE_MKLDNN=${USE_MKL} \
      USE_MKLDNN_CBLAS=${USE_MKL} \
      USE_KINETO=${USE_KINETO} \
      BUILD_TEST=0 python setup.py bdist_wheel
    status=$?
    rm -rf torch.egg-info
    pip install dist/*.whl
  else
    PYTORCH_BUILD_NUMBER=${PYTORCH_BUILD_NUMBER} \
      PYTORCH_BUILD_VERSION=${PYTORCH_BUILD_VERSION} \
      DEBUG=${DEBUG_MODE} \
      USE_ASAN=${ASAN_MODE} \
      USE_STATIC_MKL=${USE_STATIC_MKL} \
      USE_MKL=${USE_MKL} \
      USE_MKLDNN=${USE_MKL} \
      USE_MKLDNN_CBLAS=${USE_MKL} \
      USE_KINETO=${USE_KINETO} \
      BUILD_TEST=0 python setup.py install
    status=$?
  fi
  popd
  return $status
}

clean_pytorch() {
  echo -e "\033[34mCleaning PyTorch...\033[0m"
  pushd ${PYTORCH_PATH}
  python setup.py clean
  popd
}

clean_torch_musa() {
  echo -e "\033[34mCleaning torch_musa...\033[0m"
  pushd ${TORCH_MUSA_HOME}
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python setup.py clean
  rm -rf $CUR_DIR/build
  popd
}

build_torch_musa() {
  echo -e "\033[34mBuilding torch_musa...\033[0m"
  status=0
  pushd ${TORCH_MUSA_HOME}
  if [ $BUILD_WHEEL -eq 1 ]; then
    rm -rf dist build
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    PYTORCH_REPO_PATH=${PYTORCH_PATH} \
      DEBUG=${DEBUG_MODE} \
      USE_ASAN=${ASAN_MODE} \
      ENABLE_COMPILE_FP64=${COMPILE_FP64}  \
      USE_MCCL=${USE_MCCL} \
      USE_KINETO=${USE_KINETO} python setup.py bdist_wheel
    status=$?
    rm -rf torch_musa.egg-info
    pip install dist/*.whl
  else
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    PYTORCH_REPO_PATH=${PYTORCH_PATH} \
      DEBUG=${DEBUG_MODE} \
      USE_ASAN=${ASAN_MODE} \
      ENABLE_COMPILE_FP64=${COMPILE_FP64} \
      USE_MCCL=${USE_MCCL} \
      USE_KINETO=${USE_KINETO} python setup.py install
    status=$?
  fi
  if [ $status -ne 0 ]; then
    exit $status
  fi

  # scan and output ops list for each building
  bash ${CUR_DIR}/scripts/scan_ops.sh
  popd

  return $status
}

main() {
  # ======== install MUSA ========
  if [ ! -d ${MUSA_DIR} ] || [ -z "$(ls -A ${MUSA_DIR})" ]; then
    echo -e "\033[34mStart installing MUSA software stack, including musatoolkits/mudnn/mccl/muThrust/muSparse/muAlg ... \033[0m"
    . ${CUR_DIR}/docker/common/release/update_release_all.sh
  fi
  if [ ${UPDATE_MUSA} -eq 1 ]; then
    echo -e "\033[34mStart updating MUSA software stack to latest released version ... \033[0m"
    . ${CUR_DIR}/docker/common/release/update_release_all.sh
    exit 0
  fi
  if [ ${UPDATE_DAILY_MUSA} -eq 1 ]; then
    echo -e "\033[34mStart updating MUSA software stack to latest daily version ... \033[0m"
    . ${CUR_DIR}/docker/common/daily/update_daily_all.sh
    exit 0
  fi
  # ==============================

  if [[ ${CLEAN} -eq 1 ]] && [[ ${BUILD_TORCH} -ne 1 ]] && [[ ${BUILD_TORCH_MUSA} -ne 1 ]]; then
    clean_pytorch
    clean_torch_musa
    exit 0
  fi
  if [ ${ONLY_PATCH} -eq 1 ]; then
    apply_torch_patches
    exit 0
  fi
  if [ ${BUILD_TORCH} -eq 1 ]; then
    clone_pytorch
    if [ ${CLEAN} -eq 1 ]; then
      clean_pytorch
    fi
    apply_torch_patches
    build_pytorch
    build_pytorch_status=$?
    if [ $build_pytorch_status -ne 0 ]; then
      clean_and_build="bash build.sh -c  # Clean PyTorch/torch_musa and build"
      echo -e "\033[31mBuilding PyTorch failed, please try cleaning first before building: \033[0m"
      echo -e "\033[32m$clean_and_build \033[0m"
      exit 1
    fi
  fi
  if [ ${BUILD_TORCH_MUSA} -eq 1 ]; then
    if [ ${CLEAN} -eq 1 ]; then
      clean_torch_musa
    fi
    build_torch_musa
    build_torch_musa_status=$?
    if [ $build_torch_musa_status -ne 0 ]; then
      echo -e "\033[31mPlease try the following commands once building torch_musa is failed: \033[0m"
      echo -e "\033[32mClean PyTorch/torch_musa and build: \033[0m"
      echo "cmd1: bash build.sh -c"
      echo -e "\033[32mIf cmd1 still failed, update torch_musa to newest and build: \033[0m"
      echo "cmd2: git fetch && git rebase origin/main && bash build.sh -c"
      echo -e "\033[32mIf cmd2 still failed, update libraries and build: \033[0m"
      echo "cmd3: bash docker/common/daily/update_daily_musart.sh && bash docker/common/daily/update_daily_mudnn.sh && bash build.sh -c"
      echo -e "\033[32mIf cmd3 still failed, please check driver version on your host machine. \033[0m"
      exit 1
    fi
  fi
}

main
