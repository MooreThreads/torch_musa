#!/bin/bash
set -ex

MCCL_URL="http://oss.mthreads.com/release-rc/cuda_compatible/rc1.3.0/mccl_rc1.1.0.txz"
WORK_DIR="${PWD}"
DATE=$(date +%Y%m%d)

# parse parameters
parameters=`getopt -o h:: --long mccl_url:,help, -n "$0" -- "$@"`
[ $? -ne 0 ] && exit 1

eval set -- "$parameters"

while true;do
  case "$1" in
    --mccl_url) MCCL_URL=$2; shift 2;;
    --) shift ; break ;;
    *) exit 1 ;;
  esac
done

install_mccl() {
  if [ -d $1 ]; then
    rm -rf $1/mccl*.txz
  fi
  echo -e "\033[34mDownloading mccl.txz to $1\033[0m"
  wget --no-check-certificate $MCCL_URL -O $1/mccl.txz
  if [ -d $1/mccl ]; then
    rm -rf $1/mccl/*
  fi
  mkdir -p $1/mccl
  tar xJvf $1/mccl.txz -C $1/mccl
  INSTALL_DIR=$(dirname $(find $1/mccl -name install.sh))
  pushd $INSTALL_DIR
  sudo bash install.sh
  popd
}


mkdir -p $WORK_DIR/$DATE
install_mccl $WORK_DIR/$DATE
pushd ~
sudo rm -rf $WORK_DIR/$DATE
popd
