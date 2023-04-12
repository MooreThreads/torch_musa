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
apt-get install patch
apt-get install ccache
pip install -r requirements.txt
```

#### Set Important Environment Variables
```bash
export MUDNN_PATH=path/to/mudnn  # eg: MUDNN_PATH=/home/muDNN/build/mudnn
export MUSATOOLKITS_PATH=path/to/musa_toolkits  # defalut value is /usr/local/musa/
export LD_LIBRARY_PATH=$MUDNN_PATH/lib64:$MUSATOOLKITS_PATH/lib:$LD_LIBRARY_PATH
# if PYTORCH_REPO_PATH is not set, PyTorch-v2.0.0 will be downloaded outside this directory when building with build.sh
export PYTORCH_REPO_PATH=path/to/PyTorch source code 
```

### Building With Script
```bash
bash scripts/update_daily_mudnn.sh # update daily mudnn lib when needed
bash build.sh   # build original PyTorch and Torch_MUSA from scratch

# Some important parameters are as follows:
bash build.sh --torch  # build original PyTorch only
bash build.sh --musa   # build Torch_MUSA only
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
docker run -it --name=torch_musa_dev --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev:v0.1.6 /bin/bash
```

| Docker Tag | Description |
| ---- | --- |
| [**latest/v0.1.6**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-pytorch-dev/artifacts-tab) | update toolkits rc1.3.0 |
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

1. When a module is saved, it has to be copied to CPU first. Don't forget to call `model = model.to("cpu")`
   before calling `torch.save(model, model_path)`.

## Releases and Contributing


## The Team

## License
