#!/bin/bash
set -ex

install_on_ubuntu() {
  # Need the official toolchain repo to get alternate packages
  # a workaround for add-apt-repository fail
  # sudo apt-get install --reinstall ca-certificates
  # sudo -E add-apt-repository --update ppa:ubuntu-toolchain-r/test

  sudo add-apt-repository ppa:ubuntu-toolchain-r/test
  sudo apt-get update

  apt-get install -y gcc-$GCC_VERSION g++-$GCC_VERSION
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-"$GCC_VERSION" 50
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-"$GCC_VERSION" 50
  update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-"$GCC_VERSION" 50

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

install_on_centos() {
  yum install -y centos-release-scl devtoolset-$GCC_VERSION-gcc*
  mv /usr/bin/gcc /usr/bin/gcc-old && \
  mv /usr/bin/g++ /usr/bin/g++-old && \
  ln -s /opt/rh/devtoolset-$GCC_VERSION/root/bin/gcc /usr/bin/gcc && \
  ln -s /opt/rh/devtoolset-$GCC_VERSION/root/bin/g++ /usr/bin/g++
  yum clean all
  rm -rf /var/cache/yum/*
  rm -rf /var/lib/rpm/__db.*
}

install_gcc() {
  ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
  case "$ID" in
    ubuntu)
      install_on_ubuntu
      ;;
    centos)
      install_on_centos
      ;;
    *)
      echo "Unable to determine OS..."
      exit 1
      ;;
  esac
}

INSTALLED_GCC_VERSION=$(gcc --version | awk '/gcc/ {print $NF}' | cut -d. -f1)
if [ -n "$GCC_VERSION" ] && [ $INSTALLED_GCC_VERSION != $GCC_VERSION ]; then
  install_gcc
fi
