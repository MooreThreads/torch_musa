![Torch MUSA_Logo](https://github.mthreads.com/mthreads/torch_musa/blob/main/docs/source/img/torch_musa.png)
--------------------------------------------------------------------------------

[![Build Status](https://jenkins-aidev.mthreads.com/buildStatus/icon?job=torch_musa%2Fmain)](https://jenkins-aidev.mthreads.com/job/torch_musa/job/main/)


**torch_musa** is an extended Python package based on PyTorch. Developing **torch_musa** in a plug-in way allows **torch_musa** to be decoupled from PyTorch, which is convenient for code maintenance. Combined with PyTorch, users can take advantage of the strong power of Moore Threads graphics cards through **torch_musa**. In addition, **torch_musa** has two significant advantages:

* CUDA compatibility could be achieved in **torch_musa**, which greatly reduces the workload of adapting new operators.
* **torch_musa** API is consistent with PyTorch in format, which allows users accustomed to PyTorch to migrate smoothly to **torch_musa**.
--------------------------------------------------------------------------------

<!-- toc -->

- [Installation](#installation)
  - [From Python Package](#from-python-package)
  - [From Source](#from-source)
  - [Prerequisites](#prerequisites)
  - [Install Dependencies](#install-dependencies)
  - [Set Important Environment Variables](#set-important-environment-variables)
  - [Building With Script](#building-with-script-recommended)
  - [Building Step by Step From Source](#building-step-by-step-from-source)
  - [Docker Image](#docker-image)
    - [Docker Image for Developer](#docker-image-for-developer)
    - [Docker Image for User](#docker-image-for-user)
- [Getting Started](#getting-started)
  - [Key Changes](#key-changes)
  - [Example of Frequently Used APIs](#example-of-frequently-used-apis)
  - [Example of Inference Demo](#example-of-inference-demo)
  - [Example of Training Demo](#example-of-training-demo)
- [FAQ](#faq)

<!-- tocstop -->

## Installation

### From Python Package
- [Package Download Link](https://oss.mthreads.com:9001/buckets/ai-product/browse/ZGFpbHkvZnJhbWV3b3JrL3RvcmNoX211c2E=)

```bash
# for python3.8
pip install torch-2.0.0a0+gitc263bd4-cp38-cp38-linux_x86_64.whl
pip install torch_musa-2.0.0-cp38-cp38-linux_x86_64.whl

# for python3.9
pip install torch-2.0.0a0+gitc263bd4-cp39-cp39-linux_x86_64.whl
pip install torch_musa-2.0.0-cp39-cp39-linux_x86_64.whl
```

### From Source

#### Prerequisites
- [MUSA ToolKit](https://github.mthreads.com/mthreads/musa_toolkit)
- [MUDNN](https://github.mthreads.com/mthreads/muDNN)
- Other Libs (including muThrust, muSparse, muAlg, muRand)
- [PyTorch Source Code](https://github.com/pytorch/pytorch/tree/v2.0.0)
- [Docker Container Toolkits](https://mcconline.mthreads.com/software)

**NOTE:** Since some of the dependent libraries are in beta and have not yet been officially released, we recommend using the [development docker](#docker-image-for-developer) provided below to compile **torch_musa**. If you really want to compile **torch_musa** in your own environment, then please contact us for additional dependencies.

#### Install Dependencies

```bash
apt-get install ccache
apt-get install libomp-11-dev
pip install -r requirements.txt
```

#### Set Important Environment Variables
```bash
export MUSA_HOME=path/to/musa_libraries(including mudnn and musa_toolkits) # defalut value is /usr/local/musa/
export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH
# if PYTORCH_REPO_PATH is not set, PyTorch-v2.0.0 will be downloaded outside this directory when building with build.sh
export PYTORCH_REPO_PATH=path/to/PyTorch source code
```

#### Building With Script (Recommended)
```bash
bash build.sh   # build original PyTorch and torch_musa from scratch

# Some important parameters are as follows:
bash build.sh --torch  # build original PyTorch only
bash build.sh --musa   # build torch_musa only
bash build.sh --fp64   # compile fp64 in kernels using mcc in torch_musa
bash build.sh --debug  # build in debug mode
bash build.sh --asan   # build in asan mode
bash build.sh --clean  # clean everything built
```

#### Building Step by Step From Source
0. Apply PyTorch patches
```bash
bash build.sh --patch
```

1. Building PyTorch
```bash
cd pytorch
pip install -r requirements.txt
python setup.py install
# debug mode: DEBUG=1 python setup.py install
# asan mode:  USE_ASAN=1 python setup.py install
```

2. Building torch_musa
```bash
cd torch_musa
pip install -r requirements.txt
python setup.py install
# debug mode: DEBUG=1 python setup.py install
# asan mode:  USE_ASAN=1 python setup.py install
```

### Docker Image

**NOTE:** If you want to use **torch_musa** in docker container, please install [mt-container-toolkit](https://mcconline.mthreads.com/software/1?id=1) first and use '--env MTHREADS_VISIBLE_DEVICES=all' when starting a container.

#### Docker Image for Developer
```bash
docker run -it --privileged --pull always --network=host --name=torch_musa_dev --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev:latest /bin/bash
```
<details>
<summary>Docker Image List</summary>

| Docker Tag | Description |
| ---- | --- |
| [**latest/v0.1.17**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-pytorch-dev/artifacts-tab) | musatoolkits dev1.4.2 (ddk musa_2.2.0)<br> mudnn daily20230822; mccl dev1.2.2 <br> libomp-11-dev <br> muAlg_dev-0.3.0 <br> muRAND_dev1.0.0 <br> muSPARSE_dev0.1.0 <br> muThrust_dev-0.3.0 |
| [**v0.1.16**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-pytorch-dev/artifacts-tab) | build and install torchvision from source code [a6dea86] <br> other components are the same with v0.1.15 |
| [**v0.1.15**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-pytorch-dev/artifacts-tab) | musatoolkits rc1.4.0 (ddk musa_2.1.1)<br> mudnn rtm_2.1.1; mccl 20230627 <br> libomp-11-dev <br> muAlg _dev-0.1.1 <br> muRAND_dev1.0.0 <br> muSPARSE_dev0.1.0 <br> muThrust_dev-0.1.1 |
| [**v0.1.14**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-pytorch-dev/artifacts-tab) | musatoolkits rc1.4.0-rc1 (ddk rc1.4.0-rc1)<br> mudnn rc1.4.0-rc1; mccl_rc1.1.0 <br> libomp-11-dev <br> muAlg _dev-0.1.1 <br> muRAND_dev1.0.0 <br> muSPARSE_dev0.1.0 <br> muThrust_dev-0.1.1 |
| [**v0.1.13**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-pytorch-dev/artifacts-tab) | musatoolkits-20230605 (ddk_20230604 develop or newer)<br> mudnn 20230605; mccl_rc1.1.0 <br> libomp-11-dev <br> muAlg _dev-0.1.1 <br> muRAND_dev1.0.0 <br> muSPARSE_dev0.1.0 <br> muThrust_dev-0.1.1 |

</details>  

#### Docker Image for User
```bash
docker run -it --privileged --pull always --network=host --name=torch_musa_release --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g sh-harbor.mthreads.com/mt-ai/musa-pytorch-release:v0.1.12 /bin/bash
```
<details>
<summary>Docker Image List</summary>

| Docker Tag | Description |
| ---- | --- |
| [**v0.1.15**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-pytorch-release/artifacts-tab) | musatoolkits-rc1.4.0 (ddk_musa_2.1.1)<br> mudnn_rtm_2.1.1; mccl_20230627 <br> muAlg _dev-0.1.1 <br> muRAND_dev1.0.0 <br> muSPARSE_dev0.1.0 <br> muThrust_dev-0.1.1 |
| [**v0.1.12**](https://sh-harbor.mthreads.com/harbor/projects/20/repositories/musa-pytorch-release/artifacts-tab) | musatoolkits-20230605 (ddk_20230604 develop or newer)<br> mudnn 20230605; mccl_rc1.1.0 <br> muAlg _dev-0.1.1 <br> muRAND_dev1.0.0 <br> muSPARSE_dev0.1.0 <br> muThrust_dev-0.1.1 |

</details>  

## Getting Started
### Key Changes
The following two key changes are required when using **torch_musa**:
 - Import **torch_musa** package
   ```python
   import torch
   import torch_musa
   ```

 - Change the device to **musa**
   ```python
   import torch
   import torch_musa

   a = torch.tensor([1.2, 2.3], dtype=torch.float32, device='musa')
   b = torch.tensor([1.2, 2.3], dtype=torch.float32, device='cpu').to('musa')
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
  

### Example of Frequently Used APIs

<details>
<summary>code</summary>

```python
import torch
import torch_musa

torch.musa.is_available()
torch.musa.device_count()
torch.musa.synchronize()

with torch.musa.device(0):
    assert torch.musa.current_device() == 0

if torch.musa.device_count() > 1:
    torch.musa.set_device(1)
    assert torch.musa.current_device() == 1
    torch.musa.synchronize("musa:1")

a = torch.tensor([1.2, 2.3], dtype=torch.float32, device='musa')
b = torch.tensor([1.8, 1.2], dtype=torch.float32, device='musa')
c = a + b
```
</details>

### Example of Inference Demo

<details>
<summary>code</summary>

```python
import torch
import torch_musa
import torchvision.models as models

model = models.resnet50().eval()
x = torch.rand((1, 3, 224, 224), device="musa")
model = model.to("musa")
# Perform the inference
y = model(x)
```
</details>

### Example of Training Demo

<details>
<summary>code</summary>

```python
import torch
import torch_musa
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1. prepare dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 4
train_set = torchvision.datasets.CIFAR10(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)
test_set = torchvision.datasets.CIFAR10(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device("musa")

# 2. build network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net().to(device)

# 3. define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. train
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print('Finished Training')
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
net.load_state_dict(torch.load(PATH))

# 5. test
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
```
</details>

## FAQ
For more detailed information, please refer to the files in the [docs folder](https://github.mthreads.com/mthreads/torch_musa/tree/main/docs).
