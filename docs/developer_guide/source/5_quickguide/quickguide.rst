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
  b = torch.tensor([1.8, 1.2], dtype=torch.float32, device='musa')
  c = a + b
  
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
