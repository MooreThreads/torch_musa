#!/bin/bash
set -e

MINICONDA_FILE="Miniconda3-py310_23.11.0-2-Linux-x86_64.sh"
MINICONDA_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/${MINICONDA_FILE}"

mkdir -p /opt
cd /opt && \
wget --no-check-certificate ${MINICONDA_URL} && \
chmod +x ${MINICONDA_FILE} && \
mkdir -p /opt/conda && ./${MINICONDA_FILE} -b -f -p "/opt/conda" && rm -rf ${MINICONDA_FILE}
# sudo sed -e 's|PATH="\(.*\)"|PATH="/opt/conda/bin:\1"|g' -i /etc/environment
