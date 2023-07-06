## Build docker image 
1. build dev docker: [optional]You can specify the path of the root directory of pytorch on your host via `PYTORCH_REPO_ROOT_PATH` if `pytorch` repo exists.  
    ```shell
    bash build.sh -i torch_musa_docker                  \
                  --sys ubuntu:20.04                    \
                  -v 3.8                                \
                  -m ${MUSA_TOOLKITS_URL}               \
                  -n ${MUDNN_URL}
    ```
2. build release docker:
    ```shell
    bash build.sh -i torch_musa_docker                       \
                  --sys ubuntu:20.04                         \
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

## Difference between xxx.dev and xxx.release
- xxx.dev: Primarily for developers of `torch_musa`, the built docker image through `.dev` dockerfile contains development environment of `torch_musa`, including source code
- xxx.release: Primarily for users of `torch_musa`, the built docker image through `.release` dockerfile only contains the `torch` and `torch_musa` installed in the python environment.
  
If you want to install other python packages that rely on `pytorch` such as `torchvision` via pip, you should run `pip install --no-deps torchvision`, without the `--no-deps` option, it may uninstalled the `torch` that already installed in the container.

## Supported platform
- Ubuntu20.04

## Future work
- support other ubuntu system
- support centos
