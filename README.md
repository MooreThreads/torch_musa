![Torch MUSA_Logo](docs/images/torch_musa.png)
--------------------------------------------------------------------------------

**torch_musa** is an extended Python package based on PyTorch. Developing **torch_musa** in a plug-in way allows **torch_musa** to be decoupled from PyTorch, which is convenient for code maintenance. Combined with PyTorch, users can take advantage of the strong power of Moore Threads graphics cards through **torch_musa**. In addition, **torch_musa** has two significant advantages:

* CUDA compatibility could be achieved in **torch_musa**, which greatly reduces the workload of adapting new operators.
* **torch_musa** API is consistent with PyTorch in format, which allows users accustomed to PyTorch to migrate smoothly to **torch_musa**.

**torch_musa** also provides a bundle of tools for users to conduct cuda-porting, building musa extension and debugging. Please refer to [README.md](torch_musa/utils/README.md).

--------------------------------------------------------------------------------

<!-- toc -->

- [Installation](#installation)  
  - [Prerequisites](#prerequisites)
  - [Docker Image](#docker-image)
  - [From Python wheels](#from-python-wheels)
  - [From Source](#from-source)
  - [torchvision and torchaudio](#torchvision-and-torchaudio)
- [Getting Started](#getting-started)
  - [Key Changes](#key-changes)
  - [Codegen](#codegen)

<!-- tocstop -->

## Installation
### Prerequisites
Before installing torch_musa, here are some software packages that need to be installed on your machine first:
- MUSA-Driver, Container-Toolkit, MTML, sGPU, [download link](https://mcconline.mthreads.com/software)
- MUSA SDK including musa_toolkit, muDNN and MCCL, [download link](https://developer.mthreads.com/sdk/download/musa?equipment=&os=&driverVersion=&version=)
- MathLibs, including [muThrust](https://github.com/MooreThreads/muThrust) and [muAlg](https://github.com/MooreThreads/muAlg)

### Docker Image
We provide several released docker images and they can be easily found [HERE](https://mcconline.mthreads.com/repo).

During its initial startup, `Docker` performs a self-check.

```bash
# For example, start a S80 docker with Python3.10
docker run -it --privileged \
  --pull always --network=host \
  --env MTHREADS_VISIBLE_DEVICES=all \
  registry.mthreads.com/mcconline/musa-pytorch-release-public:rc4.2.0-v2.1.0-S80-py310 /bin/bash
```

### From Python wheels
Download torch_musa & PyTorch wheel from our [release page](https://github.com/MooreThreads/torch_musa/releases) and make sure you have
all prerequisites installed.

### From Source
First, clone the torch_musa repository
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

To building torch_musa, run:
```bash
bash build.sh -c  # clean cache then build PyTorch and torch_musa from scratch
```

Some important building parameters are as follows:
 - --torch/-t  # build original PyTorch only
 - --musa/-m   # build torch_musa only
 - --debug  # build in debug mode
 - --asan   # build in asan mode
 - --clean/-c  # clean everything built and build
 - --wheel/-w  # generate wheels

For example, if one has built PyTorch and only needs to build torch_musa with wheel, run:
```bash
bash build.sh -m -w
```

### torchvision and torchaudio
PyTorch v2.5.0 needs `torchvision==0.20.0` and `torchaudio==2.5.0`, and for torch_musa users we
shouldn't have them installed like `pip install torchvision==0.20.0`, instead, we should build
them from source:
```shell
# build & install torchvision
git clone https://github.com/pytorch/vision.git -b v0.20.0 --depth 1
cd visoin && python setup.py install

# build & install torchaudio
git clone https://github.com/pytorch/audio.git -b v2.5.0 --depth 1
cd audio && python setup.py install
```

## Getting Started
### Key Changes
The following two key changes are required when using **torch_musa**:
 - Import **torch_musa** package
   ```Python
   import torch
   import torch_musa
   ```

 - Change the device to **musa**
   ```Python
   import torch
   import torch_musa

   a = torch.tensor([1.2, 2.3], dtype=torch.float32, device='musa')
   b = torch.tensor([1.2, 2.3], dtype=torch.float32, device='cpu').to('musa')
   c = torch.tensor([1.2, 2.3], dtype=torch.float32).musa()
   ```
**torch musa** has integrated torchvision ops in the musa backend. Please do the following if torchvision is not installed:
- Install torchvision package via building from source
  ```
  # ensure torchvision is not installed
  pip uninstall torchvision
  
  git clone https://github.com/pytorch/vision.git
  cd vision
  python setup.py install
  ```
- Use torchvision musa backend:
  ```
  import torch
  import torch_musa
  import torchvision

  def get_forge_data(num_boxes):
      boxes = torch.cat((torch.rand(num_boxes, 2), torch.rand(num_boxes, 2) + 10), dim=1)
      assert max(boxes[:, 0]) < min(boxes[:, 2])  # x1 < x2
      assert max(boxes[:, 1]) < min(boxes[:, 3])  # y1 < y2
      scores = torch.rand(num_boxes)
      return boxes, scores

  num_boxes = 10
  boxes, scores = get_forge_data(num_boxes)
  iou_threshold = 0.5
  print(torchvision.ops.nms(boxes=boxes.to("musa"), scores=scores.to("musa"), iou_threshold=iou_threshold))
  ```

### Codegen
In torch_musa, we provide the codegen module to implement bindings and registrations of customized MUSA kernels, see [link](tools/codegen/README.md).

## License
torch_musa has a BSD-style license, as found in the [LICENSE](LICENSE) file.
