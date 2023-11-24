#!/bin/bash --login
# This script is used for daily whl release
set -exo pipefail

FILE_DIR=$(cd "$(dirname "$0")"; pwd)
TORCH_MUSA_DIR=$(dirname "${FILE_DIR}")
# ARTIFACTS_DIR as shared volume
ARTIFACTS_DIR="/artifacts/"
PYTORCH_REPO_PATH=${PYTORCH_REPO_PATH:-"/home/pytorch"}

BUILD=${BUILD_ARTIFACTS:-0}
PUBLISH=${PUBLISH_ARTIFACTS:-0}


build_artifacts() {
    git config --global --add safe.directory "*"
    # Add some description
    echo "commit id: $(git rev-parse HEAD)" > ${ARTIFACTS_DIR}README.txt
    mudnn_abs_dir=$(find $PWD -name release_mudnn*)
    mudnn_timestamp=$(find $mudnn_abs_dir -name "*.txt" | awk -F/ '{print $NF}' | awk -F_ '{print $1}')
    echo "mudnn:${mudnn_timestamp}" >> ${ARTIFACTS_DIR}README.txt
    cat $PWD/.musa_dependencies >> ${ARTIFACTS_DIR}README.txt
    source activate base

    # Build wheel packages under python3.8, using the existing conda environment
    /opt/conda/condabin/conda run -n py38 --no-capture-output USE_STATIC_MKL=1 /bin/bash build.sh -c -w

    # Move built wheel packages to shared directory ${ARTIFACTS_DIR}
    mv dist/*.whl ${ARTIFACTS_DIR} && mv ${PYTORCH_REPO_PATH}/dist/*.whl ${ARTIFACTS_DIR}

    # The py38 build cache needs to be cleaned
    /opt/conda/condabin/conda remove -y --name py38 --all

    # Build wheel packages under python3.9, create a new conda environment
    /opt/conda/condabin/conda env create -f docker/common/conda-env-torch_musa-py39.yaml
    /opt/conda/condabin/conda run -n py39 --no-capture-output pip install -r docker/common/requirements-py39.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    /opt/conda/condabin/conda run -n py39 --no-capture-output USE_STATIC_MKL=1 /bin/bash build.sh -c -w

    mv dist/*.whl ${ARTIFACTS_DIR} && mv ${PYTORCH_REPO_PATH}/dist/*.whl ${ARTIFACTS_DIR}
    /opt/conda/condabin/conda remove -y --name py39 --all

    # Build wheel packages under python3.10, create a new conda environment
    /opt/conda/condabin/conda create -y -n py310 python==3.10
    # The py39 build cache needs to be cleaned
    /opt/conda/condabin/conda run -n py310 --no-capture-output USE_STATIC_MKL=1 /bin/bash build.sh -c -w
    mv dist/*.whl ${ARTIFACTS_DIR} && mv ${PYTORCH_REPO_PATH}/dist/*.whl ${ARTIFACTS_DIR}

    
}


publish_artifacts() {
    # the build and release are done in separate containers, but they belong to the same POD
    git config --global --add safe.directory "*"
    mudnn_abs_dir=$(find $PWD -name release_mudnn*)
    mudnn_timestamp=$(find $mudnn_abs_dir -name "*.txt" | awk -F/ '{print $NF}' | awk -F_ '{print $1}')

    oss-release ${ARTIFACTS_DIR}

    # # NB: Treat the mudnn that have passed daily release as stable one
    # stable_mudnn_oss_dir="myoss/mt-ai-data/infra/framework/torch_musa/dependency/mudnn"
    # mc cp -q ${mudnn_abs_dir}/mudnn.tar ${stable_mudnn_oss_dir}/history/${mudnn_timestamp}/
    # mc cp -q ${mudnn_abs_dir}/mudnn.tar ${stable_mudnn_oss_dir}/
}

pushd ${TORCH_MUSA_DIR}
if [ ${BUILD} -eq 1 ]; then
  build_artifacts
fi
if [ ${PUBLISH} -eq 1 ]; then
  publish_artifacts
fi
popd
