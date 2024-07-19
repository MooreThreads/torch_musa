#!/bin/bash
set -ex

MKL_URL="http://oss.mthreads.com/mt-ai-data/infra/mkl-library/2023.1.0.tar"

install_mkl() {
  if [ -d $1 ]; then
    rm -rf $1/mkl*.tar
  fi
  echo -e "\033[34mDownloading mkl.tar to $1\033[0m"
  wget --no-check-certificate $MKL_URL -O $1/mkl.tar
  if [ -d $1/mkl ]; then
    rm -rf $1/mkl/*
  fi
  mkdir -p $1/mkl
  tar -xvf $1/mkl.tar -C $1/mkl --strip-components 1
  rm $1/mkl*.tar
}

mkdir -p /opt/intel/oneapi
install_mkl /opt/intel/oneapi
