![Torch MUSA_Logo](https://github.mthreads.com/mthreads/torch_musa/blob/main/docs/source/img/torch_musa.png)
--------------------------------------------------------------------------------

[![Build Status](https://jenkins-aidev.mthreads.com/buildStatus/icon?job=torch_musa%2Fmain)](https://jenkins-aidev.mthreads.com/job/torch_musa/job/main/)


Musa PyTorch is an extension that adds Moore Threads's MUSA as a standalone PyTorch backend

<!-- toc -->

- [More About Torch_MUSA](#more-about-torch_musa)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Install Dependencies](#install-dependencies)
  - [Set Important Environment Variables](#set-important-environment-variables)
  - [Building With Script](#building-with-script)
  - [Building Step by Step From Source](#building-step-by-step-from-source)
  - [Docker Image](#docker-image)
- [Getting Started](#getting-started)
- [Releases and Contributing](#releases-and-contributing)
- [The Team](#the-team)
- [License](#license)

<!-- tocstop -->

## More About Torch_MUSA


## Installation

#### Prerequisites
- [MUSA ToolKit](https://github.mthreads.com/mthreads/musa_toolkit)
- [MUDNN](https://github.mthreads.com/mthreads/muDNN)
- [PyTorch Source Code](https://github.com/pytorch/pytorch/tree/v2.0.0)

#### Install Dependencies

```bash
apt-get install ccache
pip install -r requirements.txt
```

#### Set Important Environment Variables
```bash
export MUSA_HOME=path/to/musa_libraries(including mudnn and musa_toolkits) # defalut value is /usr/local/musa/
export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH
# if PYTORCH_REPO_PATH is not set, PyTorch-v2.0.0 will be downloaded outside this directory when building with build.sh
export PYTORCH_REPO_PATH=path/to/PyTorch source code
```

### Building With Script
```bash
bash scripts/update_daily_mudnn.sh # update daily mudnn lib if needed
bash build.sh   # build original PyTorch and Torch_MUSA from scratch

# Some important parameters are as follows:
bash build.sh --torch  # build original PyTorch only
bash build.sh --musa   # build Torch_MUSA only
bash build.sh --fp64   # compile fp64 in kernels using mcc in Torch_MUSA
bash build.sh --debug  # build in debug mode
bash build.sh --asan   # build in asan mode
bash build.sh --clean  # clean everything built
```

### Building Step by Step From Source
- 0.Apply PyTorch patches
```bash
bash build.sh --only-patch
```

- 1.Building PyTorch
```bash
cd pytorch
pip install -r requirements.txt
python setup.py install
# debug mode: DEBUG=1 python setup.py install
# asan mode:  USE_ASAN=1 python setup.py install
```

- 2.Building Torch_MUSA
```bash
cd torch_musa
pip install -r requirements.txt
python setup.py install
# debug mode: DEBUG=1 python setup.py install
# asan mode:  USE_ASAN=1 python setup.py install
```

### Docker Image

```bash
docker run -it --name=torch_musa_dev --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev:v0.1.7 /bin/bash
```

| Docker Tag | Description |
| ---- | --- |
| [**latest/v0.1.7**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-pytorch-dev/artifacts-tab) | toolkits rc1.3.0 + MUSA-Runtime_use_armory<br> muAlg _dev-0.1.0 <br> muRAND_dev1.0.0 <br> muSPARSE_dev0.1.0 <br> muThrust_dev-0.1.0 |
| [**v0.1.6**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-pytorch-dev/artifacts-tab) | toolkits rc1.3.0 |
| [**v0.1.5**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-pytorch-dev/artifacts-tab) |  |

## Getting Started
```bash
import torch
import torch_musa

a = torch.tensor([1.2, 2.3], dtype=torch.float32, device='musa')
b = torch.tensor([1.8, 1.2], dtype=torch.float32, device='musa')
c = a + b
```

### Limitations
Though a user can access most PyTorch features seamlessly with torch_musa, there are several
features temporarily unavailable.
...

## Releases and Contributing


## The Team

## License
