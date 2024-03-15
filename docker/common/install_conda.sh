#!/bin/bash
set -e

MINICONDA_VERSION="latest"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh"

pushd /tmp
wget --no-check-certificate $MINICONDA_URL -O conda_install.sh
mkdir -p /opt/conda
sudo bash ./conda_install.sh -b -f -p "/opt/conda"
sudo sed -e 's|PATH="\(.*\)"|PATH="/opt/conda/bin:\1"|g' -i /etc/environment
sudo rm -f ./conda_install.sh
popd
