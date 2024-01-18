.. note::
   | 使用torch_musa时，需要先导入torch包（import torch）和torch_musa包(import torch_musa)。

常用环境变量
--------------

开发torch_musa过程中常用环境变量如下表所示：

.. tabularcolumns:: |m{0.45\textwidth}|m{0.10\textwidth}|m{0.35\textwidth}|
.. table:: 常用环境变量

   +-------------------------------------+-----------+---------------------------------------------------------------------+
   |              环境变量示例           | 所属组件  |          功能说明                                                   |
   +=====================================+===========+=====================================================================+
   | export TORCH_SHOW_CPP_STACKTRACES=1 |  PyTorch  | 当python程序发生错误时显示PyTorch中C++调用栈                        |
   +-------------------------------------+-----------+---------------------------------------------------------------------+
   | export MUDNN_LOG_LEVEL=INFO         |   MUDNN   | 使能MUDNN算子库调用的log                                            |
   +-------------------------------------+-----------+---------------------------------------------------------------------+
   | export MUSA_VISIBLE_DEVICES=0,1,2,3 |   Driver  | 控制当前可见的显卡序号                                              |
   +-------------------------------------+-----------+---------------------------------------------------------------------+
   | export MUSA_LAUNCH_BLOCKING=1       |   Driver  | 驱动以同步模式下发kernel，即当前kernel执行结束后再下发下一个kernel  |
   +-------------------------------------+-----------+---------------------------------------------------------------------+


常用api示例代码
-----------------------

.. code-block:: python

  import torch
  import torch_musa
  
  torch.musa.is_available()
  torch.musa.device_count()
  
  a = torch.tensor([1.2, 2.3], dtype=torch.float32, device='musa')
  b = torch.tensor([1.8, 1.2], dtype=torch.float32, device='cpu').to('musa')
  c = torch.tensor([1.8, 1.3], dtype=torch.float32).musa()
  d = a + b + c
  
  torch.musa.synchronize()
  
  with torch.musa.device(0):
      assert torch.musa.current_device() == 0
  
  if torch.musa.device_count() > 1:
      torch.musa.set_device(1)
      assert torch.musa.current_device() == 1
      torch.musa.synchronize("musa:1")

torch_musa中python api基本与PyTorch原生api接口保持一致，极大降低了新用户的学习成本。

推理示例代码
-------------

.. code-block:: python

  import torch
  import torch_musa
  import torchvision.models as models
  
  model = models.resnet50().eval()
  x = torch.rand((1, 3, 224, 224), device="musa")
  model = model.to("musa")
  # Perform the inference
  y = model(x)


训练示例代码
-------------

.. code-block:: python

  import torch
  import torch_musa
  import torchvision
  import torchvision.transforms as transforms
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim
  
  ## 1. prepare dataset
  transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  
  batch_size = 4
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           shuffle=False, num_workers=2)
  classes = ('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  device = torch.device("musa")
  
  ## 2. build network
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
  
  ## 3. define loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  
  ## 4. train
  for epoch in range(2):  # loop over the dataset multiple times
      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
  
          # zero the parameter gradients
          optimizer.zero_grad()
  
          # forward + backward + optimize
          outputs = net(inputs.to(device))
          loss = criterion(outputs, labels.to(device))
          loss.backward()
          optimizer.step()
  
          # print statistics
          running_loss += loss.item()
          if i % 2000 == 1999:    # print every 2000 mini-batches
              print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
              running_loss = 0.0
  
  print('Finished Training')
  
  PATH = './cifar_net.pth'
  torch.save(net.state_dict(), PATH)
  
  net.load_state_dict(torch.load(PATH))
  
  ## 5. test
  correct = 0
  total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
      for data in testloader:
          images, labels = data
          # calculate outputs by running images through the network
          outputs = net(images.to(device))
          # the class with the highest energy is what we choose as prediction
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels.to(device)).sum().item()
  
  print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


混合精度AMP训练示例代码
-----------------------

.. code-block:: python

  import torch
  import torch_musa
  import torch.nn as nn

  class SimpleModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.fc1 = nn.Linear(5, 4)
          self.relu = nn.ReLU()
          self.fc2 = nn.Linear(4, 3)
  
      def forward(self, x):
          x = self.fc1(x)
          x = self.relu(x)
          x = self.fc2(x)
          return x
      def __call__(self, x):
          return self.forward(x)
  
  DEVICE = "musa"
  
  def train_in_amp(low_dtype=torch.float16):
      model = SimpleModel().to(DEVICE)
      criterion = nn.MSELoss()
      optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
  
      # create the scaler object
      scaler = torch.musa.amp.GradScaler()
  
      inputs = torch.randn(6, 5).to(DEVICE)  # 将数据移至GPU
      targets = torch.randn(6, 3).to(DEVICE)
      for step in range(20):
          optimizer.zero_grad()
          # create autocast environment
          with torch.musa.amp.autocast(dtype=low_dtype):
              outputs = model(inputs)
              assert outputs.dtype == low_dtype
              loss = criterion(outputs, targets)
  
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()
      return loss

  if __name__ == "__main__":
      train_in_amp(torch.float16)


分布式训练示例代码
------------------

.. code-block:: python

  """Demo of DistributedDataParall"""
  import os
  import torch
  from torch import nn
  from torch import optim
  from torch.nn.parallel import DistributedDataParallel as DDP
  import torch.distributed as dist
  import torch.multiprocessing as mp
  import torch_musa
  
  
  class Model(nn.Module):
      def __init__(self):
          super().__init__()
          self.linear = nn.Linear(5,5)
      def forward(self, x):
          return self.linear(x)
  
  def start(rank, world_size):
      if os.getenv("MASTER_ADDR") is None:
          os.environ["MASTER_ADDR"]= ip # IP must be specified here
      if os.getenv("MASTER_PORT") is None:
          os.environ["MASTER_PORT"]= port # port must be specified here
      dist.init_process_group("mccl", rank=rank, world_size=world_size)
  
  def clean():
      dist.destroy_process_group()
  
  def runner(rank, world_size):
      torch_musa.set_device(rank)
      start(rank, world_size)
      model = Model().to('musa')
      ddp_model = DDP(model, device_ids=[rank])
      optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
      for _ in range(5):
          input_tensor = torch.randn(5, dtype=torch.float, requires_grad=True).to('musa')
          target_tensor = torch.zeros(5, dtype=torch.float).to('musa')
          output_tensor = ddp_model(input_tensor)
          loss_f = nn.MSELoss()
          loss = loss_f(output_tensor, target_tensor)
          loss.backward()
          optimizer.step()
      clean()
  
  if __name__ == "__main__":
      mp.spawn(runner, args=(2,), nprocs=2, join=True)


使能TensorCore示例代码
----------------------

在s4000上，当输入数据类型是flaot32时，可以通过设置TensorFloat32来使能TensorCore，从而加速计算过程。TensorFloat32的加速原理可以参考 `TensorFloat-32 <https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices>`_ 。

.. code-block:: python

  import torch
  import torch_musa
  with torch.backends.mudnn.flags(allow_tf32=True):
      assert torch.backends.mudnn.allow_tf32
      a = torch.randn(10240, 10240, dtype=torch.float, device='musa')
      b = torch.randn(10240, 10240, dtype=torch.float, device='musa')
      result_tf32 = a @ b

  torch.backends.mudnn.allow_tf32 = True
  assert torch_musa._MUSAC._get_allow_tf32()
  a = torch.randn(10240, 10240, dtype=torch.float, device='musa')
  b = torch.randn(10240, 10240, dtype=torch.float, device='musa')
  result_tf32 = a @ b


C++部署示例代码
---------------

.. code-block:: cpp

  #include <torch/script.h>
  #include <torch_musa/csrc/core/Device.h>
  #include <iostream>
  #include <memory>

  int main(int argc, const char* argv[]) {
    // Register 'musa' for PrivateUse1 as we save model with 'musa'.
    c10::register_privateuse1_backend("musa");

    torch::jit::script::Module module;
    // Load model which saved with torch jit.trace or jit.script.
    module = torch::jit::load(argv[1]);

    std::vector<torch::jit::IValue> inputs;
    // Ready for input data.
    torch::Tensor input = torch::rand({1, 3, 224, 224}).to("musa");
    inputs.push_back(input);

    // Model execute.
    at::Tensor output = module.forward(inputs).toTensor();

    return 0;
  }

详细用法请参考 `examples/cpp <https://github.mthreads.com/mthreads/torch_musa/tree/main/examples/cpp>`_ 下内容。
