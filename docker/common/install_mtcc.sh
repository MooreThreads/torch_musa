#!/bin/bash
set -ex

MTCC_URL="https://oss.mthreads.com/release-ci/computeQA/musa/newest/mtcc-nightly-x86_64-linux-gnu-ubuntu-20.04.tar.gz"
WORK_DIR="${PWD}"
DATE=$(date +%Y%m%d)

# parse parameters
parameters=`getopt -o h:: --long mtcc_url:,help, -n "$0" -- "$@"`
[ $? -ne 0 ] && exit 1

eval set -- "$parameters"

while true;do
  case "$1" in
    --mtcc_url) MTCC_URL=$2; shift 2;;
    --) shift ; break ;;
    *) exit 1 ;;
  esac
done

install_mtcc() {
  if [ -d $1 ]; then
    rm -rf $1/mtcc*.tar.gz
  fi
  echo -e "\033[34mDownloading mtcc.tar.gz to $1\033[0m"
  wget --no-check-certificate $MTCC_URL -O $1/mtcc.tar.gz
  if [ -d $1/mtcc ]; then
    rm -rf $1/mtcc/*
  fi
  mkdir -p $1/mtcc
  tar zxvf $1/mtcc.tar.gz -C $1/mtcc
  INSTALL_DIR=$(dirname $(find $1/mtcc -name install.sh))
  pushd $INSTALL_DIR
  sudo bash install.sh
  popd
}


mkdir -p $WORK_DIR/$DATE
install_mtcc $WORK_DIR/$DATE
pushd ~
rm -rf $WORK_DIR/$DATE
popd
