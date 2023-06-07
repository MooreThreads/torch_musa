#!/bin/bash
set -ex

install_ubuntu() {
  # Install common dependencies
  apt-get update
  apt-get install -y tzdata
  ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
  dpkg-reconfigure --frontend noninteractive tzdata

  ccache_deps="asciidoc docbook-xml docbook-xsl xsltproc"
  apt-get install -y --no-install-recommends \
    sudo \
    git \
    curl \
    wget \
    vim \
    gdb \
    ssh \
    $ccache_deps \
    make \
    cmake \
    autoconf \
    automake \
    clang \
    libclang-dev \
    libelf-dev \
    gcc-multilib \
    build-essential \
    llvm \
    libelf1 \
    patch \
    ca-certificates \
    software-properties-common \
    libtool \
    unzip \
    gawk \
    bison \
    expect \
    libnuma-dev

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

install_centos() {
  echo "centos is unsupported yet."
  exit 1
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  centos)
    install_centos
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
