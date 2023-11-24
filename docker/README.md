<!-- toc -->
- [Introduction](#Introduction)
- [Codebase Structure](#codebase-structure)
- [Build Docker Image](#Build-docker-image)
    - [Build Development Docker image](#Build-development-docker-image)
        - [Step1: Build Base Docker Image](#step1-build-base-docker-image)
        - [Step2: Build Development Docker Image](#step2-build-development-docker-image)
    - [Build Release Docker Image](#Build-release-docker-image)
    - [Build All Docker Images](#Build-all-docker-images)
- [Run Docker Container](#Run-docker-container)
- [Supported Platform](#Supported-platform)
- [Supported Python Version](#Supported-Python-version)
- [Future Work](#Future-work)
## Introduction
There are two versions of **torch_musa** docker images: development, and release. For development docker image, **PyTorch** and **torch_musa** are installed from the source code, and the source code of both are in the /home directory. For release docker image, **PyTorch** and **torch_musa** are installed directly using the whl package in the Python virtual environment.
<br>Note:If you want to install other Python packages that rely on **PyTorch** such as **torchvision** via pip, you should run `pip install --no-deps torchvision`. Because it may uninstalled the **PyTorch** that already installed in the container  without the `--no-deps` option.

## Codebase Structure
- [common](./common/) - Contains some common files required for building docker, such as install_gcc.sh, etc.
- [ubuntu](./ubuntu/) - Contains the dockerfile file for building the ubuntu operating system.
- [build_base.sh](./build_base.sh) - Script used to build base docker image that is the first step to build development docker image.
- [build.sh](./build.sh) - Script used to build development and release docker image.
- [build_all.sh](./build_all.sh) - Script used to build different versions of development and release docker image at once.

## Build Docker Image 
### Build Development Docker Image
Follow the two steps below to build the development docker image.
#### Step1: Build Base Docker Image
Use docker/build_base.sh to build the base docker image and the base docker image contains the **PyTorch** source code and other basic packages, such as gdb, ccache. You can specify the path of the root directory of **PyTorch** on your host machine via 
**PYTORCH_REPO_ROOT_PATH** if **PyTorch** repo exists or where **PyTorch** will be downloaded to.
#### Parameters of build_base.sh
* `-n`/`--name`：Name of the docker image, default:NULL.
* `-t`/`--tag`：Tag of the docker image, default:latest.
* `-s`/`--sys`：The operating system, for example ubuntu:20.04, now only support ubuntu, defualt:ubuntu:20.04.
* `-v`/`--python_version`：The Python version used by **torch_musa**, default:3.8.
* `-h`/`--help`：Help information.

    ```shell
    DOCKER_BUILD_DIR=/tmp/torch_musa_base_docker_build \
    PYTORCH_REPO_ROOT_PATH=~/tmp \
    
    #build base docker image for Python3.10
    bash docker/build_base.sh -n pytorch2.0.0_py310 -s ubuntu:20.04 -v 3.10
    
    #build base docker image for Python3.9
    bash docker/build_base.sh -n pytorch2.0.0_py39 -s ubuntu:20.04 -v 3.9
    
    #build base docker image for Python3.8
    bash docker/build_base.sh -n pytorch2.0.0_py38 -s ubuntu:20.04 -v 3.8
    ```

#### Step2: Build Development Docker Image
After using the docker/build_base.sh script to construct the base docker image, specified within the dockerfile.dev, the musa_toolkit, muDNN, and mccl will be installed upon the base docker image. Then **PyTorch** and **torchvision** will be installed. Lastly, **torch_musa** will be installed and subsequently tested, with the test results saved in /home/integration_test_output.txt and /home/ut_output.txt.
You can specify the path of the root directory of **PyTorch** on your host machine via **TORCH_VISION_REPO_ROOT_PATH** if **vision** repo exists or where **torchvision** will be download to. The dockerfile.dev installs **PyTorch** using torch_musa/build.sh -t, so please make sure that the **PYTORCH_TAG** in build_bash.sh matches the **PYTORCH_TAG** in torch_musa/build.sh. Also, ensure that the **VISION_TAG** in build.sh is compatible with **PyTorch**.
#### Parameters of build.sh
* `-n`/`--name`：Name of the docker image, default:NULL.
* `-t`/`--tag`：Tag of the docker image, default:latest.
* `-b`/`--base_img`：The base docker image, for example ubuntu:20.04. When building the development Doceker image, select the base docker image you built in the previous step. If you are building the release docker image, select the base image such as ubuntu:20.04. default:NULL.
* `-f`/`--docker_file`：The path of dockerfile, options are dockerfile.release and dockerfile.dev, default:"".
* `-v`/`--python_version`：The Python version used by **torch_musa**, now support 3.8, 3.9 and 3.10, default:3.8. This option is only useful when building release docker image, and when building development docker image, it defaults to the Python environment of the base docker image.
* `-m`/`--musa_toolkits_url`：The download link of MUSA ToolKit, default:"".
* `--mudnn_url`：The download link of MUDNN, default:"".
* `--mccl_url`：The download link of MUSA MCCL, default:"".
* `-r`/`--release`：The build pattern, please set it when dockerfile is dockerfile.release, default:unset.
* `--torch_whl_url`：The download link of **PyTorch** wheel, this option is only useful when building release docker image, default:"".
* `--torch_musa_whl_url`：The download link of **torch_musa** wheel, this option is only useful when building release docker image, default:"".
* `--torch_musa_tag`：The tag of **torch_musa** to be installed, this option is only useful when building development docker image, default:"dev1.5.1".
* `--no_prepare`：Whether to prepare build context, such as **torch_musa**. Set it when using build_all.sh to build docker image because build_all.sh will prepare context, default:unset.

    ```shell
    DOCKER_BUILD_DIR=/data/torch_musa_docker_build \
    TORCH_VISION_REPO_ROOT_PATH=/tmp \
    bash docker/build.sh -n torch_musa_dev \
                         -b sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev-py39:base-pytorch2.0.0 \
                         -f docker/ubuntu/dockerfile.dev \
                         -m ${MUSA_TOOLKITS_URL} \
                         --mudnn_url ${MUDNN_URL} \
                         --mccl_url ${MCCL_URL} \
                         --torch_musa_tag ${TORCH_MUSA_TAG}
    ```
You can explicit specify the context of **docker build** through **DOCKER_BUILD_DIR**. 
### Build Release Docker Image
Please refer to the previous section for the meaning of each parameter


```shell
bash build.sh -n torch_musa_docker                       \
              -b ubuntu:20.04                            \
              -f ./ubuntu/dockerfile.release             \
              -v 3.8                                     \
              -m ${MUSA_TOOLKITS_URL}                    \
              --mudnn_url ${MUDNN_URL}                   \
              --mccl_url ${MCCL_URL}                     \
              --torch_whl_url ${TORCH_WHL_URL}           \
              --torch_musa_whl_url ${TORCH_MUSA_WHL_URL} \
              --release
```  

Please run `bash build.sh -h` to see the specific meaning of the parameters.  

## Build All Docker Images
Build development docker image and release docker image with different Python version in one step using build_all.sh. Note that downloading url should be specified **manually** in build_all.sh.

```shell
cd docker && bash build_all.sh
```

## Run Docker Container
```shell
docker run -itd --privileged=true                  \
                --env MTHREADS_VISIBLE_DEVICES=all \
                --name ${CONTAINER_NAME}           \
                --shm-size 80G ${IMAGE_NAME} /bin/bash
docker start ${CONTAINER_NAME}
docker exec -it ${CONTAINER_NAME} /bin/bash
```
  


## Supported Platform
- Ubuntu20.04

## Supported Python Version
- py3.8 
- py3.9 
- py3.10

## Future Work
- support other ubuntu systems
- support centos
