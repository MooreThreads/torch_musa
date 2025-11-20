---
title: 编译安装
description: torch_musa 编译安装
hide_table_of_contents: False
---

# 编译安装

:::warning 注意:

编译安装前，需要安装 MUSAToolkits 软件包，MUDNN 库，MCCL 库，muThrust 库，muAlg库，muRAND 库，muSPARSE 库。具体安装步骤，请参见相应组件的安装手册。

:::

## 依赖环境

- Python == 3.8/3.9/3.10。

- 摩尔线程 MUSA 软件包，推荐版本如下：
  
  - MUSA 驱动 2.7.0‑rc4‑0822
  
  - MUSAToolkits rc3.1.0
  
  - MUDNN rc2.7.0
  
  - MCCL rc1.7.0
  
  - [muAlg](https://github.com/MooreThreads/muAlg)
  
  - [mutlass](https://github.com/MooreThreads/mutlass)
  
  - [muThrust](https://github.com/MooreThreads/muThrust)
  
  - [Docker Container Toolkits](https://mcconline.mthreads.com/software)

## 编译流程

1. 向 PyTorch 源码打 patch

2. 编译 PyTorch

3. 编译 torch_musa

torch_musa rc1.3.0 是在 PyTorch v2.2.0 基础上以插件的方式来支持摩尔线程显卡。开发时涉及到对PyTorch 源码的修改，目前是以打 patch 的方式实现的。PyTorch 社区正在积极支持第三方后端入,这个 [issue](https://github.com/pytorch/pytorch/issues/98406) 下有相关 PR。我们也在积极向 PyTorch 社区提交 PR，避免在编译过程中向 PyTorch 打 patch。

## 开发 Docker

为了方便开发者开发 torch_musa，我们提供了开发用的 docker image，参考命令：

```shell
docker run -it --name=torch_musa_dev --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev-py38:latest /bin/bash
```

:::warning 注意:

使用 docker 时，请务必提前安装 [mt‑container‑toolkit](https://mcconline.mthreads.com/software/1?id=1) ，并且在启动 docker container 时添加选项'‑‑env MTHREADS_VISIBLE_DEVICES=all'，否则在docker container内部无法使用torch_musa。

:::

## 编译步骤

### 设置环境变量

```shell
export MUSA_HOME=path/to/musa_libraries(including musa_toolkits, mudnn and so on) # defalut value is /usr/local/musa/
export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH
export PYTORCH_REPO_PATH=path/to/PyTorch source code
# if PYTORCH_REPO_PATH is not set, PyTorch-v2.0.0 will be downloaded outside this directory automatically when building with build.sh
```

### 使用脚本一键编译（推荐）

```shell
cd torch_musa
bash docker/common/daily/update_daily_mudnn.sh # update daily mudnn lib if needed
bash build.sh # build original PyTorch and torch_musa from scratc# Some important parameters are as follows:
bash build.sh --torch # build original PyTorch only
bash build.sh --musa # build torch_musa only
bash build.sh --fp64 # compile fp64 in kernels using mcc in torch_musa
bash build.sh --debug # build in debug mode
bash build.sh --asan # build in asan mode
bash build.sh --clean # clean everything built
```

在初次编译时，需要执行 `bash build.sh` （先编译 PyTorch，再编译 torch_musa）。在后续开发过程中，如果不涉及对 PyTorch 源码的修改，那么执行` bash build.sh -m `（仅编译 torch_musa）即可。

### 分步骤编译

如果不想使用脚本编译，那么可以按照如下步骤逐步编译。

1. 在 PyTorch 打 patch

```shell
# 请保证 PyTorch 源码和 torch_musa 源码在同级目录或者 export PYTORCH_REPO_PATH=path/to/PyTorch 指向 PyTorch 源码
bash build.sh --only-patch
```

2. 编译 PyTorch

```shell
cd pytorch
pip install -r requirements.txt
python setup.py install
# debug mode: DEBUG=1 python setup.py install
# asan mode: USE_ASAN=1 python setup.py install
```

3. 编译 torch_musa

```shell
cd torch_musa
pip install -r requirements.txt
python setup.py install
# debug mode: DEBUG=1 python setup.py install
# asan mode: USE_ASAN=1 python setup.py install
```
