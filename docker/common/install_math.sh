#!/bin/bash
set -ex

ARCH=GPU_ARCH_MP_21 # default to MP_21 arch for muBLAS
MT_OPENCV_URL="http://oss.mthreads.com/release-ci/Math-X/mt_opencv.tar.gz"
# use this mu_rand_url in next docker image version
# MU_RAND_URL="https://oss.mthreads.com/release-ci/Math-X/muRAND_dev1.0.0.tar.gz"
MU_RAND_URL="https://oss.mthreads.com/release-ci/computeQA/mathX/newest/murand.tar.gz"
MU_SPARSE_URL="http://oss.mthreads.com/release-ci/Math-X/muSPARSE_dev0.1.0.tar.gz"
MU_ALG_URL="https://oss.mthreads.com/release-ci/computeQA/mathX/newest/mualg.tar"
MU_THRUST_URL="https://oss.mthreads.com/release-ci/computeQA/mathX/newest/muthrust.tar"
MU_BLAS_URL="https://oss.mthreads.com/release-ci/computeQA/mathX/newest/${ARCH}/mublas.tar.gz"

WORK_DIR="${PWD}"
DATE=$(date +%Y%m%d)

# parse parameters
parameters=$(getopt -o h:: --long mt_opencv_url:,mu_rand_url:,mu_sparse_url:,mu_alg_url:,mu_thrust_url:,mu_blas_url:,help, -n "$0" -- "$@")
[ $? -ne 0 ] && exit 1

eval set -- "$parameters"

while true; do
  case "$1" in
  --mt_opencv_url)
    MT_OPENCV_URL=$2
    shift 2
    ;;
  --mu_rand_url)
    MU_RAND_URL=$2
    shift 2
    ;;
  --mu_sparse_url)
    MU_SPARSE_URL=$2
    shift 2
    ;;
  --mu_alg_url)
    MU_ALG_URL=$2
    shift 2
    ;;
  --mu_thrust_url)
    MU_THRUST_URL=$2
    shift 2
    ;;
  --mu_blas_url)
    MU_BLAS_URL=$2
    shift 2
    ;;
  --)
    shift
    break
    ;;
  *) exit 1 ;;
  esac
done

install_mu_rand() {
  if [ -d $1 ]; then
    rm -rf $1/mu_rand*.tar.gz
  fi
  echo -e "\033[34mDownloading mu_rand.tar.gz to $1\033[0m"
  wget --no-check-certificate $MU_RAND_URL -O $1/mu_rand.tar.gz
  if [ -d $1/muRand ]; then
    rm -rf $1/muRand/*
  fi
  mkdir -p $1/muRand
  tar -zxf $1/mu_rand.tar.gz -C $1/muRand
  INSTALL_DIR=$(dirname $(find $1/muRand -name install.sh))
  pushd $INSTALL_DIR
  sudo bash install.sh
  popd
}

install_mu_sparse() {
  if [ -d $1 ]; then
    rm -rf $1/mu_sparse*.tar.gz
  fi
  echo -e "\033[34mDownloading mu_sparse.tar.gz to $1\033[0m"
  wget --no-check-certificate $MU_SPARSE_URL -O $1/mu_sparse.tar.gz
  if [ -d $1/muSparse ]; then
    rm -rf $1/muSparse/*
  fi
  mkdir -p $1/muSparse
  tar -zxf $1/mu_sparse.tar.gz -C $1/muSparse
  INSTALL_DIR=$(dirname $(find $1/muSparse -name install.sh))
  pushd $INSTALL_DIR
  sudo bash install.sh
  popd
}

install_mu_alg() {
  suffix=$(basename "$MU_ALG_URL" | awk -F. '{print $NF}')
  if [ -d $1 ]; then
    rm -rf $1/mu_alg*.deb
    rm -rf $1/mu_alg*.tar
  fi
  if [ ${suffix} == "tar" ]; then
    echo -e "\033[34mDownloading mu_alg.tar to $1\033[0m"
    wget --no-check-certificate $MU_ALG_URL -O $1/mu_alg.tar
    mkdir -p $1/muAlg
    tar xf $1/mu_alg.tar --strip-components 2 -C $1/muAlg
    pushd $1/muAlg
    ls | xargs dpkg -i
    popd
  elif [ ${suffix} == "deb" ]; then
    echo -e "\033[34mDownloading mu_alg.deb to $1\033[0m"
    wget --no-check-certificate $MU_ALG_URL -O $1/mu_alg.deb
    sudo dpkg -i $1/mu_alg.deb
  fi
}

install_thrust() {
  suffix=$(basename "$MU_THRUST_URL" | awk -F. '{print $NF}')
  if [ -d $1 ]; then
    rm -rf $1/mu_thrust*.deb
    rm -rf $1/mu_thrust.tar
  fi
  if [ ${suffix} == "tar" ]; then
    echo -e "\033[34mDownloading mu_thrust.tar to $1\033[0m"
    wget --no-check-certificate $MU_THRUST_URL -O $1/mu_thrust.tar
    mkdir -p $1/muThrust
    tar xf $1/mu_thrust.tar --strip-components 2 -C $1/muThrust
    pushd $1/muThrust
    ls | xargs dpkg -i
    popd
  elif [ ${suffix} == "deb" ]; then
    echo -e "\033[34mDownloading mu_thrust.deb to $1\033[0m"
    wget --no-check-certificate $MU_THRUST_URL -O $1/mu_thrust.deb
    sudo dpkg -i $1/mu_thrust.deb
  fi
}

install_blas() {
  if [ -d $1 ]; then
    rm -rf $1/mublas*.tar.gz
  fi
  echo -e "\033[34mDownloading mublas.tar.gz to $1\033[0m"
  wget --no-check-certificate $MU_BLAS_URL -O $1/mublas.tar.gz
  extracted="$1/muBLAS"
  if [ -d extracted ]; then
    rm -rf "$extracted*"
  fi
  mkdir -p $extracted
  tar -zxf $1/mublas.tar.gz -C $extracted
  INSTALL_DIR=$(dirname $(find $extracted -name install.sh))
  pushd $INSTALL_DIR
  sudo bash install.sh
  popd
}

main() {
  # Get all install function names
  function_names=$(grep "^install" $0 | sed -nE 's/^([a-zA-Z0-9_]+)\(.*/\1/p')
  mkdir -p $WORK_DIR/$DATE
  for fn_name in $function_names; do
    eval $fn_name $WORK_DIR/$DATE
  done
  pushd ~
  rm -rf $WORK_DIR/$DATE
  popd
}

main
