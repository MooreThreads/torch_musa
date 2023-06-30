![Torch MUSA_Logo](https://github.com/MooreThreads/torch_musa/blob/main/docs/source/img/torch_musa.png)

[中文](https://github.com/MooreThreads/torch_musa/blob/main/README_zh.md)    [English](https://github.com/MooreThreads/torch_musa/blob/main/README.md)

--------------------------------------------------------------------------------

**torch_musa** 是一个基于PyTorch的扩展。 以插件形式开发 **torch_musa** 可以将 **torch_musa** 与 PyTorch 解耦, 便于代码维护。 结合 PyTorch, 用户可以通过 **torch_musa** 使用摩尔线程显卡的强大功能。 此外, **torch_musa** 还具有两个优势:

* **torch_musa** 兼容 CUDA, 这将极大减少适配的工作量。
* **torch_musa** API 在格式上 与 PyTorch 保持一致, 可以让用户平滑的从 PyTorch 迁移到 **torch_musa**.
--------------------------------------------------------------------------------

<!-- toc -->

- [安装](#安装)
  - [从 Python Package](#从-Python-Package)
  - [从源码安装](#从源码安装)
  - [前置环境](#前置环境)
  - [安装依赖](#安装依赖)
  - [设置关键环境变量](#设置关键环境变量)
  - [从脚本构建](#从脚本构建-推荐)
  - [从源代码逐步构建](#从源代码逐步构建)
  - [Docker 镜像](#Docker-镜像)
    - [开发者镜像](#开发者镜像)
    - [用户镜像](#用户镜像)
- [入门](#入门)
  - [关键变更](#关键变更)
  - [常用API示例](#常用API示例)
  - [推理演示示例](#推理演示示例)
  - [训练演示示例](#训练演示示例)
- [文档](#文档)
- [FAQ](#faq)

<!-- tocstop -->

## 安装

### 从 Python Package

```bash
# for python3.8
pip install torch-2.0.0a0+gitc263bd4-cp38-cp38-linux_x86_64.whl
pip install torch_musa-2.0.0-cp38-cp38-linux_x86_64.whl

# for python3.9
pip install torch-2.0.0a0+gitc263bd4-cp39-cp39-linux_x86_64.whl
pip install torch_musa-2.0.0-cp39-cp39-linux_x86_64.whl
```

### 从源码安装

#### 前置环境
- MUSA ToolKit
- MUDNN
- 其他库 (including muThrust, muSparse, muAlg, muRand)
- [PyTorch 源码](https://github.com/pytorch/pytorch/tree/v2.0.0)

**注:** 由于一些库处于 beta 状态尚未正式发布，我们建议使用下面提供的 [开发者镜像](#开发者镜像) 来编译 **torch_musa**。 如果你确实想要在自己的环境中编译 **torch_musa** 请联系我们获取附加的依赖。

#### 安装依赖

```bash
apt-get install ccache
pip install -r requirements.txt
```

#### 设置关键环境变量
```bash
export MUSA_HOME=path/to/musa_libraries(including mudnn and musa_toolkits) # 默认值是 /usr/local/musa/
export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH
# 如果 PYTORCH_REPO_PATH 没有设置, 在使用 build.sh 构建时，PyTorch-v2.0.0 会被下载到该目录外。
export PYTORCH_REPO_PATH=path/to/PyTorch source code
```

#### 从脚本构建 (推荐)
```bash
bash build.sh   # 从头开始构建 PyTorch 和 torch_musa

# 下面是一些重要的参数:
bash build.sh --torch  # 只构建原版 PyTorch
bash build.sh --musa   # 只构建 torch_musa
bash build.sh --fp64   # compile fp64 in kernels using mcc in torch_musa
bash build.sh --debug  # debug 模式在构建
bash build.sh --asan   # asan 模式下构建
bash build.sh --clean  # 清理所有构建
```

#### 从源代码逐步构建
0. 应用 PyTorch patches
```bash
bash build.sh --only-patch
```

1. 构建 PyTorch
```bash
cd pytorch
pip install -r requirements.txt
python setup.py install
# debug 模式: DEBUG=1 python setup.py install
# asan 模式:  USE_ASAN=1 python setup.py install
```

2. 构建 torch_musa
```bash
cd torch_musa
pip install -r requirements.txt
python setup.py install
# debug 模式: DEBUG=1 python setup.py install
# asan 模式:  USE_ASAN=1 python setup.py install
```

### Docker 镜像
#### 开发者镜像
该部分很快支持。

#### 用户镜像
该部分很快支持。

## 入门
### 关键变更
使用 **torch_musa** 时需要进行以下两项关键更改:
 - 导入 **torch_musa** 包
   ```python
   import torch
   import torch_musa
   ```

 - 将 device 改为 **musa**
   ```python
   import torch
   import torch_musa

   a = torch.tensor([1.2, 2.3], dtype=torch.float32, device='musa')
   b = torch.tensor([1.2, 2.3], dtype=torch.float32, device='cpu').to('musa')
   ```

### 常用API示例

<details>
<summary>Code</summary>

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

### 推理演示示例

<details>
<summary>Code</summary>

```python
import torch
import torch_musa
import torchvision.models as models

model = models.resnet50().eval()
x = torch.rand((1, 3, 224, 224), device="musa")
model = model.to("musa")
# 进行推理
y = model(x)
```
</details>

### 训练演示示例

<details>
<summary>Code</summary>

```python
import torch
import torch_musa
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1. 准备数据集
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

# 2. 构建网络
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

# 3. 定义 loss 值和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. 训练
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

# 5. 测试
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

## 文档
- [开发指南](https://github.com/MooreThreads/torch_musa/blob/main/docs/MooreThreads-Torch_MUSA-Developer-Guide-CN-v1.0.0.pdf)

## FAQ
更多信息请参考 [docs folder](https://github.com/MooreThreads/torch_musa/tree/main/docs) 目录下的文件。
