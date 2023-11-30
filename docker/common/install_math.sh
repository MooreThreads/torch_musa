#!/bin/bash
# Please note: mublas and murand have been included in musa_toolkit
set -e

GPU=$(mthreads-gmi -q -i 0 | grep "Product Name" | awk -F: '{print $2}' | tr -d '[:space:]')
ARCH="GPU_ARCH_MP_21"
if [ "$GPU" = "MTTS3000" ] || [ "$GPU" = "MTTS80" ]; then
    ARCH="GPU_ARCH_MP_21"
elif [ "$GPU" = "MTTS4000" ] || [ "$GPU" = "MTTS90" ]; then
    ARCH="GPU_ARCH_MP_22"
else
    echo -e "\033[31mThe output of mthreads-gmi -q -i 0 | grep \"Product Name\" | awk -F: '{print \$2}' | tr -d '[:space:]' is not correct! Now GPU ARCH is set to qy1 by default! \033[0m"
fi

WORK_DIR="${PWD}"
DATE=$(date +%Y%m%d)

do_install_mtopencv="false"
do_install_murand="false"
do_install_musparse="false"
do_install_mualg="false"
do_install_muthrust="false"
do_install_mublas="false"

MT_OPENCV_URL="http://oss.mthreads.com/release-ci/Math-X/mt_opencv.tar.gz"
MU_RAND_URL="https://oss.mthreads.com/release-ci/computeQA/mathX/newest/${ARCH}/murand.tar.gz"
MU_SPARSE_URL="http://oss.mthreads.com/release-ci/Math-X/muSPARSE_dev0.1.0.tar.gz"
MU_ALG_URL="https://oss.mthreads.com/release-ci/computeQA/mathX/newest/mualg.tar"
MU_THRUST_URL="https://oss.mthreads.com/release-ci/computeQA/mathX/newest/muthrust.tar"
MU_BLAS_URL="https://oss.mthreads.com/release-ci/computeQA/mathX/newest/${ARCH}/mublas.tar.gz"

echo_info() {
  echo -e "\033[33m"$1"\033[0m"
}

echo_success() {
  echo -e "\033[32m"$1"\033[0m"
}

help() {
  echo_info "----------------------------------------------------------------"
  name="$(basename "$(realpath "${BASH_SOURCE:-$0}")")"
  echo_info "Description:"
  echo_info "This script will install math libs of MUSA,"
  echo_info "including muRand, muSparse, muAlg, muThrust, muBLAS."
  echo_info "Please note: mublas and murand have been included in musa_toolkit"
  echo_info "Usage:"
  echo_info " ${name} [-w] [-c] [-r] [-s] [-a] [-t] [-b]"
  echo_info "Details:"
  echo_info " -w : install all the math libs"
  echo_info " -c : only install mt-opencv"
  echo_info " -r : only install muRand"
  echo_info " -s : only install muSparse"
  echo_info " -a : only install muAlg"
  echo_info " -t : only install muThrust"
  echo_info " -b : only install muBLAS"
  echo_info " -h : print help message"
  echo_info "----------------------------------------------------------------"
  echo_info "e.g."
  echo_info "${name} -c # install mtOpenCV"
  exit 0
}

# parse parameters

while getopts 'wcrsatbh:' OPT; do
  case $OPT in
  c)
    do_install_mtopencv="true"
    ;;
  r)
    do_install_murand="true"
    ;;
  s)
    do_install_musparse="true"
    ;;
  a)
    do_install_mualg="true"
    ;;
  t)
    do_install_muthrust="true"
    ;;
  b)
    do_install_mublas="true"
    ;;
  w)
    do_install_mtopencv="true"
    do_install_murand="true"
    do_install_musparse="true"
    do_install_mualg="true"
    do_install_muthrust="true"
    do_install_mublas="true"
    ;;
  h) help ;;
  ?) help ;;
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
  echo_info "install math libs of arch: ${ARCH}"
  [ ! -d "$WORK_DIR/$DATE" ] && mkdir -p "$WORK_DIR/$DATE"
  # FIXME:(mingyuan.wang) mtOpenCV now doesn't included
  [ "${do_install_murand}" = "true" ] && install_mu_rand "$WORK_DIR/$DATE"
  [ "${do_install_musparse}" = "true" ] && install_mu_sparse "$WORK_DIR/$DATE"
  [ "${do_install_mualg}" = "true" ] && install_mu_alg "$WORK_DIR/$DATE"
  [ "${do_install_muthrust}" = "true" ] && install_thrust "$WORK_DIR/$DATE"
  [ "${do_install_mublas}" = "true" ] && install_blas "$WORK_DIR/$DATE"
  rm -rf $WORK_DIR/$DATE
}

main