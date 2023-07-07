![Torch MUSA_Logo](https://github.com/MooreThreads/torch_musa/blob/main/docs/source/img/torch_musa.png)
--------------------------------------------------------------------------------

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
- [Documentation](#documentation)
- [FAQ](#faq)

<!-- tocstop -->

## Installation

### From Python Package
- [Package Download Link](https://github.com/MooreThreads/torch_musa/releases)

```bash
# for python3.8
pip install torch-2.0.0_xxxxxx-cp38-cp38-linux_x86_64.whl
pip install torch_musa_xxxxxx-cp38-cp38-linux_x86_64.whl

# for python3.9
pip install torch-2.0.0_xxxxxx-cp39-cp39-linux_x86_64.whl
pip install torch_musa_xxxxxx-cp39-cp39-linux_x86_64.whl
```

### From Source

#### Prerequisites
- [MUSA ToolKit](https://new-developer.mthreads.com/sdk/download/musa?equipment=&os=&driverVersion=&version=)
- [MUDNN](https://new-developer.mthreads.com/sdk/download/musa?equipment=&os=&driverVersion=&version=)
- Other Libs (including muThrust, muSparse, muAlg, muRand)
- [PyTorch Source Code](https://github.com/pytorch/pytorch/tree/v2.0.0)
- Docker Container Toolkits

**NOTE:** Since some of the dependent libraries are in beta and have not yet been officially released, we recommend using the [development docker](#docker-image-for-developer) provided below to compile **torch_musa**. If you really want to compile **torch_musa** in your own environment, then please contact us for additional dependencies.

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
- [Development Docker Image Download Link](https://mcconline.mthreads.com/repo/musa-pytorch-dev-public?repoName=musa-pytorch-dev-public&repoNamespace=mcconline&displayName=MUSA%20Pytorch%20Dev%20Public)
- [Release Docker Image Download Link](https://mcconline.mthreads.com/repo/musa-pytorch-release-public?repoName=musa-pytorch-release-public&repoNamespace=mcconline&displayName=MUSA%20Pytorch%20Release%20Public)
#### Docker Image for Developer

```bash
docker run -it --privileged --name=torch_musa_dev --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g torch_musa_develop_image /bin/bash
```
<details>
<summary>Docker Image List</summary>

| Docker Tag | Description |
| ---- | --- |
| [**latest/v1.0.0**](https://mcconline.mthreads.com/repo/musa-pytorch-dev-public?repoName=musa-pytorch-dev-public&repoNamespace=mcconline&displayName=MUSA%20Pytorch%20Dev%20Public) | musatoolkits rc1.4.0 (requires musa driver musa_2.1.1)<br> mudnn rtm_2.1.1; mccl 20230627 <br> libomp-11-dev <br> muAlg _dev-0.1.1 <br> muRAND_dev1.0.0 <br> muSPARSE_dev0.1.0 <br> muThrust_dev-0.1.1 |

</details>  


#### Docker Image for User
```bash
docker run -it --privileged --name=torch_musa_release --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g torch_musa_release_image /bin/bash
```
<details>
<summary>Docker Image List</summary>

| Docker Tag | Description |
| ---- | --- |
| [**latest/v1.0.0**](https://mcconline.mthreads.com/repo/musa-pytorch-release-public?repoName=musa-pytorch-release-public&repoNamespace=mcconline&displayName=MUSA%20Pytorch%20Release%20Public) | musatoolkits rc1.4.0 (requires musa driver musa_2.1.1)<br> mudnn rtm_2.1.1; mccl 20230627 <br> libomp-11-dev <br> muAlg _dev-0.1.1 <br> muRAND_dev1.0.0 <br> muSPARSE_dev0.1.0 <br> muThrust_dev-0.1.1 |

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

## Documentation
- [Developer Guide](https://github.com/MooreThreads/torch_musa/blob/main/docs/MooreThreads-Torch_MUSA-Developer-Guide-CN-v1.0.0.pdf)

## FAQ
For more detailed information, please refer to the files in the [docs folder](https://github.com/MooreThreads/torch_musa/tree/main/docs). Please let us know by email **developers@mthreads.com** if you have any questions.
