![Torch MUSA_Logo](docs/pdf/images/torch_musa.png)
--------------------------------------------------------------------------------

<!-- toc -->

- [Overview](#overview)
- [Usage](#usage)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Docker Image](#docker-image)
  - [From Python wheels](#from-python-wheels)
  - [From Source](#from-source)
- [MUSA Supported Repositories](#musa-supported-repositories)
  - [torchvision](#torchvision)
  - [torchaudio](#torchaudio)
  - [Other Repositories](#other-repositories)
- [License](#license)

<!-- tocstop -->

## Overview

**torch_musa** is an extended Python package based on PyTorch. Combined with PyTorch, users can take advantage of the strong power of Moore Threads graphics cards through **torch_musa**.

**torch_musa**'s APIs are consistent with PyTorch in format, which allows users accustomed to PyTorch to migrate smoothly to **torch_musa**, so for the usage users can refer to [PyTorch Official Doc](https://docs.pytorch.org/docs/stable/index.html), all you need is just switch the backend string from `cpu` or `cuda` to `musa`.

**torch_musa** also provides a bundle of tools for users to conduct cuda-porting, building musa extension and debugging. Please refer to [README.md](torch_musa/utils/README.md).

For some customize optimizations, like **Dynamic Double Casting** and **Unified Memory Management**, please refer to [README.md](torch_musa/README.md).

If you want to write your layers in C/C++, we provide a convenient extension API that is efficient and with minimal boilerplate. No wrapper code needs to be written. Refer to [ResNet50 example](torch_musa/examples/cpp/README.md).

--------------------------------------------------------------------------------

## Usage

**We recommend users refer to [torchada](https://github.com/MooreThreads/torchada), which enable your torch scripts written with CUDA run directly on our MUSA platform.**

Now `import torch` would automatically load torch_musa, and in most cases, one just need to switch the backend from **cuda** to **musa**:
```Python
import torch
torch.musa.is_available()  # should be True

# Creating tensors:
a = torch.tensor([1.2, 2.3], dtype=torch.float32, device='musa')
b = torch.tensor([1.2, 2.3], dtype=torch.float32, device='cpu').to('musa')
c = torch.tensor([1.2, 2.3], dtype=torch.float32).musa()

# Also some cuda modules or functions:
torch.backends.mudnn.allow_tf32 = True
event = torch.musa.Event()
stream = torch.musa.Stream()
```

For distribute training or inference, initialize your process group with backend **mccl**:
```Python
import torch.distributed as dist

dist.init_process_group("mccl", rank=rank, world_size=world_size)
```

## Installation
### Prerequisites
Before installing torch_musa, here are some software packages that need to be installed on your machine first:
- (For Docker users) **[KUAE Cloud Native Toolkits](https://developer.mthreads.com/sdk/download/CloudNative?equipment=&os=&driverVersion=&version=)**, including Container-Toolkit, MTML, sGPU;
- **[MUSA-SDK](https://developer.mthreads.com/sdk/download/musa?equipment=&os=&driverVersion=&version=)**, including MUSA Driver, musa_toolkit, muDNN and MCCL(*S4000 only*);
- **Other libraries**, including [muThrust](https://github.com/MooreThreads/muThrust) and [muAlg](https://github.com/MooreThreads/muAlg)

### Docker Image
We provide several released docker images and they can be easily found in [mcconline](https://mcconline.mthreads.com/repo).

Pull the docker image with `docker pull`, create a container with command below and there you go.
```bash
# For example, start a S80 docker with Python3.10
docker run -it --privileged \
  --pull always --network=host \
  --env MTHREADS_VISIBLE_DEVICES=all \
  registry.mthreads.com/mcconline/musa-pytorch-release-public:latest /bin/bash
```
During its initial startup, it performs a self-check, so you can see the MUSA environment is ready or not.

### From Python wheels
Download torch & torch_musa wheels from our [Release Page](https://github.com/MooreThreads/torch_musa/releases),
please make sure you have all prerequisites installed.

### From Source
Firstly, clone the torch_musa repository
```bash
git clone https://github.com/MooreThreads/torch_musa.git
cd torch_musa
```

then, we need to set `MUSA_HOME`, `LD_LIBRARY_PATH` and `PYTORCH_REPO_PATH` for building torch_musa:
```bash
export MUSA_HOME=path/to/musa_libraries(including mudnn and musa_toolkits) # defalut value is /usr/local/musa/
export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH
# if PYTORCH_REPO_PATH is not set, PyTorch will be downloaded outside this directory when building with build.sh
export PYTORCH_REPO_PATH=/path/to/PyTorch
```

To build torch_musa, run:
```bash
bash build.sh -c  # clean cache then build PyTorch and torch_musa from scratch
```

Some important building parameters are as follows:
 - --torch/-t: build original PyTorch only
 - --musa/-m: build torch_musa only
 - --debug: build in debug mode
 - --asan: build in asan mode
 - --clean/-c: clean everything built and build
 - --wheel/-w: generate wheels

For example, if one has built PyTorch and only needs to build torch_musa and generate wheel, run:
```bash
bash build.sh -m -w
```
For **S80/S3000** users, since the MCCL library is not provided for such architectures, please add `USE_MCCL=0` whilst building torch_musa:
```bash
USE_MCCL=0 bash build.sh -c
```

## MUSA Supported Repositories

### torchvision
For torch_musa v2.7.0 and later, install torchvision from [our repository](https://gitub.com/MooreThreads/vision):
```shell
git clone https://github.com/MooreThreads/vision -b v0.22.1-musa --depth 1
cd vision && python setup.py install
```

Otherwise, install torchvision from [torch repository](https://github.com/pytorch/vision):
```shell
git clone https://github.com/pytorch/vision -b ${version} --depth 1
cd vision && python setup.py install
```
the `version` depends on torch version, for example you have torch v2.5.0, `${version}` should be `0.20.0`.

### torchaudio
Install torchaudio from [torch source](https://github.com/pytorch/audio):
```shell
git clone https://github.com/pytorch/audio.git -b ${version} --depth 1
cd audio && python setup.py install
```
the `version` is same as the torch version.

### Other Repositories
Many repositories have supported MUSA backend upstream,
like [Transformers](https://github.com/huggingface/transformers.git), [Accelerate](https://github.com/huggingface/accelerate.git),
you can install them from PyPi with `pip install [repo-name]`.

For others that haven't supported musa, we musified them and put into our [GitHub](https://github.com/MooreThreads), here's the list:
| Repo | Branch | Link |  Build command |
| :-- | :-: | :-: | :-: |
| pytorch3d | musa-dev | https://github.com/MooreThreads/pytorch3d | python setup.py install |
| pytorch_sparse | master | https://github.com/MooreThreads/pytorch_sparse | python setup.py install |
| pytorch_scatter | master | https://github.com/MooreThreads/pytorch_scatter | python setup.py install |
| torchvision | v0.22.1-musa | https://github.com/MooreThreads/vision | python setup.py install |
| pytorch_lightning | musa-dev | https://github.com/MooreThreads/pytorch-lightning | python setup.py install |

If users find any question about these repos, please file issues in torch_musa, and if anyone  musify a repository, you can
submit a Pull Request that helping us to expand this list.

## License
torch_musa has a BSD-style license, as found in the [LICENSE](LICENSE) file.
