## Build docker image 
1. Make sure you are under the `torch_musa/docker/` directory before you start building docker image
2. - If you want to build image in `develop mode`, save your git account and password in a credential file, an example is as follows:
       ```tex
       ${YOUR_GIT_ACCOUNT}
       ${YOUR_GIT_PASSWORD}
       ```
    - Otherwise, copy `torch_musa/requirements.txt` to `torch_musa/docker/`
3. Start to build docker image, an example to build docker is as follows:  
    ```shell
    # develop mode
    bash build.sh -i torch_musa_docker                  \
                  -c ${YOUR_GIT_CREDENTIAL_FILE_PATH}   \
                  --sys ubuntu:20.04                    \
                  --python_version 3.8

    # release mode
    bash build.sh -i torch_musa_docker                       \
                  --sys ubuntu:20.04                         \
                  --python_version 3.8                       \
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
- xxx.dev: Primarily for developers of `torch_musa`, the builded docker image through `.dev` dockerfile contains development environment of `torch_musa`, including source code
- xxx.release: Primarily for users of `torch_musa`, the builded docker image through `.release` dockerfile only contains the `torch` and `torch_musa` installed in the python environment.

## Supported platform
- Ubuntu20.04

## Future work
- support other ubuntu system
- support centos
