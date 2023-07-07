.. attention::
   | 编译安装前，需要安装MUSAToolkits软件包，MUDNN库，muThrust库，muAlg库，muRAND库，muSPARSE库。具体安装步骤，请参见相应组件的安装手册。

依赖环境
----------------------------

- Python == 3.8 或者 Python == 3.9。
- 摩尔线程MUSA软件包，推荐版本如下：

  * MUSA驱动(https://new-developer.mthreads.com/sdk/download/musa?equipment=&os=&driverVersion=&version=)
  * MUSAToolkits工具包(https://new-developer.mthreads.com/sdk/download/musa?equipment=&os=&driverVersion=&version=)
  * MUDNN算子库(https://new-developer.mthreads.com/sdk/download/musa?equipment=&os=&driverVersion=&version=)
  * muAlg_dev-0.1.1-Linux.deb
  * muRAND_dev1.0.0.tar.gz
  * muSPARSE_dev0.1.0.tar.gz
  * muThrust_dev-0.1.1-Linux.deb
  * Docker Container Toolkits



编译流程
---------

#. 向PyTorch源码打patch
#. 编译PyTorch
#. 编译torch_musa

torch_musa是在PyTorch v2.0.0基础上以插件的方式来支持摩尔线程显卡。开发时涉及到对PyTorch源码的修改，目前是以打patch的方式实现的。PyTorch社区正在积极支持第三方后端接入，https://github.com/pytorch/pytorch/issues/98406 这个issue下有相关PR。我们也在积极向PyTorch社区提交PR，避免在编译过程中向PyTorch打patch。


开发Docker镜像
-----------

为了方便开发者开发torch_musa，我们提供了开发用的docker image(https://mcconline.mthreads.com/repo/musa-pytorch-dev-public?repoName=musa-pytorch-dev-public&repoNamespace=mcconline&displayName=MUSA%20Pytorch%20Dev%20Public)，参考命令：

.. code-block:: bash

  docker run -it --name=torch_musa_dev --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g torch_musa_develop_image /bin/bash

开发docker镜像中已经安装了必需的依赖包，包括一些未正式发布的依赖包。如果用户不想在我们提供的docker镜像中开发，请通过developers@mthreads.com邮箱联系我们获取必需的依赖包。

编译步骤
---------

设置环境变量
^^^^^^^^^^^^^

.. code-block:: bash

  export MUSA_HOME=path/to/musa_libraries(including musa_toolkits, mudnn and so on) # defalut value is /usr/local/musa/
  export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH
  export PYTORCH_REPO_PATH=path/to/PyTorch source code
  # if PYTORCH_REPO_PATH is not set, PyTorch-v2.0.0 will be downloaded outside this directory automatically when building with build.sh

使用脚本一键编译（推荐）
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cd torch_musa
  bash scripts/update_daily_mudnn.sh # update daily mudnn lib if needed
  bash build.sh   # build original PyTorch and Torch_MUSA from scratch
  
  # Some important parameters are as follows:
  bash build.sh --torch  # build original PyTorch only
  bash build.sh --musa   # build Torch_MUSA only
  bash build.sh --fp64   # compile fp64 in kernels using mcc in Torch_MUSA
  bash build.sh --debug  # build in debug mode
  bash build.sh --asan   # build in asan mode
  bash build.sh --clean  # clean everything built

在初次编译时，需要执行 ``bash build.sh`` （先编译PyTorch，再编译torch_musa）。 在后续开发过程中，如果不涉及对PyTorch源码的修改，那么执行 ``bash build.sh -m`` （仅编译torch_musa）即可。

分步骤编译
^^^^^^^^^^^

如果不想使用脚本编译，那么可以按照如下步骤逐步编译。

1. 在PyTorch打patch

.. code-block:: bash

  # 请保证PyTorch源码和torch_musa源码在同级目录或者export PYTORCH_REPO_PATH=path/to/PyTorch指向PyTorch源码
  bash build.sh --patch

2. 编译PyTorch

.. code-block:: bash

  cd pytorch
  pip install -r requirements.txt
  python setup.py install
  # debug mode: DEBUG=1 python setup.py install
  # asan mode:  USE_ASAN=1 python setup.py install

3. 编译torch_musa

.. code-block:: bash

  cd torch_musa
  pip install -r requirements.txt
  python setup.py install
  # debug mode: DEBUG=1 python setup.py install
  # asan mode:  USE_ASAN=1 python setup.py install
