## Build docker image 
### 1. build dev docker
Follow the two steps below to build the dev docker image
1. **build base docker:** the base docker image contains the pytorch source code and other basic packages, such as gdb, ccache.
You can specify the path of the root directory of pytorch on your host via `PYTORCH_REPO_ROOT_PATH` if `pytorch` repo exists or where pytorch will be download to
    ```shell
    DOCKER_BUILD_DIR=/tmp/torch_musa_base_docker_build \
    PYTORCH_REPO_ROOT_PATH=~/tmp \
    bash docker/build_base.sh -i pytorch2.0.0 -s ubuntu:20.04
    ```
2. **build dev docker:** the dev docker contains the torch_musa source code and other musa softwares, such as musatooklit, mudnn.
You can specify the path of the root directory of pytorch on your host via `TORCH_VISION_REPO_ROOT_PATH` if `vision` repo exists or where torchvision will be download to
    ```shell
    DOCKER_BUILD_DIR=/data/torch_musa_docker_build \
    TORCH_VISION_REPO_ROOT_PATH=/tmp \
    bash docker/build.sh -i torch_musa_dev \
                         -b sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev:base-pytorch-v2.0.0 \
                         -f docker/ubuntu/dockerfile.dev \
                         -m ${MUSA_TOOLKITS_URL} \
                         -n ${MUDNN_URL}
    ```
You can explicit specify the context of `docker build` through `DOCKER_BUILD_DIR` 
### 2. build release docker
```shell
bash build.sh -i torch_musa_docker                       \
              -b ubuntu:20.04                            \
              -f ./ubuntu/dockerfile.release             \
              -v 3.8                                     \
              -m ${MUSA_TOOLKITS_URL}                    \
              -n ${MUDNN_URL}                            \
              --torch_whl_url ${TORCH_WHL_URL}           \
              --torch_musa_whl_url ${TORCH_MUSA_WHL_URL} \
              --release
```  

Please run `bash build.sh -h` to see the specific meaning of the parameters.  

## Run docker container
```shell
docker run -itd --privileged=true                  \
                --env MTHREADS_VISIBLE_DEVICES=all \
                --name ${CONTAINER_NAME}           \
                --shm-size 80G ${IMAGE_NAME} /bin/bash
docker start ${CONTAINER_NAME}
docker exec -it ${CONTAINER_NAME} /bin/bash
```

## Meaning of each folder
- common: place some commonly used scripts, such as `install_conda.sh`
- ubuntu: place the dockerfile that is depended on building the docker image

## Difference between xxx.base, xxx.dev and xxx.release
- xxx.base: so that we dont need to save the pytorch source code locally or download from remote every time we update the docker image.
- xxx.dev: Primarily for developers of `torch_musa`, the built docker image through `.dev` dockerfile contains development environment of `torch_musa`, including source code
- xxx.release: Primarily for users of `torch_musa`, the built docker image through `.release` dockerfile only contains the `torch` and `torch_musa` installed in the python environment.
  
If you want to install other python packages that rely on `pytorch` such as `torchvision` via pip, you should run `pip install --no-deps torchvision`, without the `--no-deps` option, it may uninstalled the `torch` that already installed in the container.

## Supported platform
- Ubuntu20.04

## Future work
- support other ubuntu system
- support centos
