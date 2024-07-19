profiler工具
-------------

可以使用torch_musa对PyTorch官方性能分析工具 (`官方示例 <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_)  **torch.profiler** 适配的版本来对模型进行性能分析，**torch_musa模块的导入必须在torch.profiler或者torch.autograd.profiler模块** 导入之前，如下是一个最佳实践:

.. code-block:: python

  import torch
  import torch_musa
  import torchvision.models as models
  from torch.profiler import profile, record_function, schedule
  import torch.nn as nn
  import torch.optim
  import torch.utils.data
  import torchvision
  import torchvision.transforms as T


  model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
  model.musa()
  transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=0)
  criterion = nn.CrossEntropyLoss().musa()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  device = torch.device("musa:0")
  model.train()

  my_schedule = schedule(
    skip_first=1,
    wait=1,
    warmup=1,
    active=2,
    repeat=2)

  def trace_handler(p):
    output = p.key_averages().table(sort_by="self_musa_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")
    
  # with_stack=True requires experimental_config's setting at torch 2.0.0 which has not resolved the issue, https://github.com/pytorch/pytorch/issues/100253
  with profile(
    schedule=my_schedule,
    on_trace_ready=trace_handler,
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
    with record_function("model_training"):
      for step, data in enumerate(trainloader, 0):
          print("step:{}".format(step))
          inputs, labels = data[0].to(device=device), data[1].to(device=device)
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          prof.step()        
          if step >= 10:
              break

  print(prof.key_averages(group_by_input_shape=True).table(sort_by="musa_time_total", row_limit=10))
  print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_musa_time_total", row_limit=2))
  # clone FlameGraph and execute `./flamegraph.pl --title "MUSA time" --countname "us." /tmp/profiler_stacks.txt > perf_viz.svg`
  prof.export_stacks("/tmp/profiler_stacks.txt", "self_musa_time_total")

上述的一个例子基本涵盖了在使用profiler分析模型在单机单卡上训练或推理情景下，各种参数使用时所导出的结果情况。其中，
 - record_shapes指定是否记录算子输入的形状
 - profile_memory指定是否记录算子执行过程中发生的内存分配和释放的总量
 - schedule指定profiler分析的调度函数，遵循skip_first->[wait->warmup->active]->[wait->warmup->active]...，[]作为一个周期，其中
    - skip_first表明跳过初始的步数
    - wait表明要等待的步数
    - warmup表明热身的步数
    - active表明正常工作的步数
    - repeat表明周期数
 - on_trace_ready指定函数来在每个周期结束后对新产生的trace进行处理
注意，在当前适配的profiler版本中，**use_musa** 和 **activities** 参数无效，这两个参数的设定不会影响profiler的配置。

单机多卡示例如下：

.. code-block:: python

  import os
  import torch
  import torch.backends.cudnn as cudnn
  import torch.distributed as dist
  import torch.multiprocessing as mp
  import torch.nn as nn
  import torch.optim
  import torch.profiler
  import torch.utils.data
  import torchvision
  import torchvision.transforms as T
  from torch.nn.parallel import DistributedDataParallel as DDP
  from torchvision import models
  import torch_musa

  def clean():
      dist.destroy_process_group()

  def example(rank, use_gpu=True):
      if use_gpu:
          torch_musa.set_device(rank)
          model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
          model.to("musa")
          cudnn.benchmark = True
          model = DDP(model, device_ids=[rank])
      else:
          model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
          model = DDP(model)

      # Use gradient compression to reduce communication
      # model.register_comm_hook(None, default.fp16_compress_hook)
      # or
      # state = powerSGD_hook.PowerSGDState(process_group=None,matrix_approximation_rank=1,start_powerSGD_iter=2)
      # model.register_comm_hook(state, powerSGD_hook.powerSGD_hook)

      transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
      trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform)
      train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
      trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=train_sampler,
                                                shuffle=False, num_workers=4)

      if use_gpu:
          criterion = nn.CrossEntropyLoss().to(rank)
      else:
          criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
      model.train()

      with torch.profiler.profile(
          activities=[
              torch.profiler.ProfilerActivity.CPU,
              torch.profiler.ProfilerActivity.MUSA],
          schedule=torch.profiler.schedule(
              wait=1,
              warmup=1,
              active=2),
          on_trace_ready=torch.profiler.tensorboard_trace_handler('./result_ddp', worker_name='worker'+str(rank)),
          record_shapes=True,
          profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
          with_stack=True,
          experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
      ) as p:
          for step, data in enumerate(trainloader, 0):
              print("step:{}".format(step))
              if use_gpu:
                  inputs, labels = data[0].to("musa"), data[1].to("musa")
              else:
                  inputs, labels = data[0], data[1]
              outputs = model(inputs)
              loss = criterion(outputs, labels)

              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              p.step()
              if step + 1 >= 10:
                  break
      clean()


  def init_process(rank, size, fn, backend='mccl'):
      """ Initialize the distributed environment. """
      os.environ['MASTER_ADDR'] = '127.0.0.1'
      os.environ['MASTER_PORT'] = '29500'
      dist.init_process_group(backend, rank=rank, world_size=size)
      fn(rank, size)


  if __name__ == "__main__":
      size = torch.musa.device_count()
      processes = []
      mp.set_start_method("spawn")
      for rank in range(size):
          p = mp.Process(target=init_process, args=(rank, size, example))
          p.start()
          processes.append(p)

      for p in processes:
          p.join()

多机多卡示例如下:

.. code-block:: python

  # Usage:
  # e.g:
  # On machine A:
  #   MASTER_ADDR=xxx MASTER_PORT=xxx python3 resnet50_distributed_ddp_profiler.py -n 2 -g 1 -nr 0
  #
  # On machine B:
  #   MASTER_ADDR=xxx MASTER_PORT=xxx python3 resnet50_distributed_ddp_profiler.py -n 2 -g 1 -nr 1

  import os
  import argparse
  import torch
  from torch import nn
  from torch import optim
  from torch.nn.parallel import DistributedDataParallel as DDP
  import torch.distributed as dist
  import torch.multiprocessing as mp
  import torch_musa
  import torchvision
  import torchvision.transforms as T
  from torchvision import models

  model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

  transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=4)
  criterion = nn.CrossEntropyLoss()

  def clean():
      dist.destroy_process_group()

  def start(rank, world_size):
      if os.getenv("MASTER_ADDR") is None:
          os.environ['MASTER_ADDR'] = '127.0.0.1'
      if os.getenv("MASTER_PORT") is None:
          os.environ['MASTER_PORT'] = '29500'
      dist.init_process_group("mccl", rank=rank, world_size=world_size)

  def runner(gpu, args):
      rank = args.nr * args.gpus + gpu
      torch_musa.set_device(rank % torch.musa.device_count())
      start(rank, args.world_size)
      model.to('musa')
      ddp_model = DDP(model, device_ids=[rank % torch.musa.device_count()])
      optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
      with torch.profiler.profile(
          activities=[
              torch.profiler.ProfilerActivity.CPU,
              torch.profiler.ProfilerActivity.MUSA],
          schedule=torch.profiler.schedule(
              wait=1,
              warmup=1,
              active=2),
          on_trace_ready=torch.profiler.tensorboard_trace_handler('./result_dist_ddp', worker_name='worker'+str(rank)),
          record_shapes=True,
          profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
          with_stack=True,
          experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
      ) as p:
          for step, data in enumerate(trainloader, 0):
              inputs, labels = data[0].to('musa'), data[1].to('musa')
              outputs = ddp_model(inputs)
              loss = criterion(outputs, labels)
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              p.step()
              if step + 1 >= 4:
                  break
      clean()

  def train(fn, args):
      mp.spawn(fn, args=(args,), nprocs=args.gpus, join=True)

  if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument('-n', '--nodes', default=1,
                          type=int, metavar='N')
      parser.add_argument('-g', '--gpus', default=1, type=int,
                          help='number of gpus per node')
      parser.add_argument('-nr', '--nr', default=0, type=int,
                          help='ranking within the nodes')
      args = parser.parse_args()
      args.world_size = args.gpus * args.nodes
      train(runner, args)


使能TensorCore优化
------------------

**s4000** 支持 **TensorFloat32(TF32)** tensor cores。利用tensor cores，可以使得 **矩阵乘** 和 **卷积** 计算得到加速。torch_musa中TF32的使用方式和CUDA中一致，可以参考 `PyTorch官方示例 <https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices>`_ 。在快速入门章节中，我们也提供了示例代码，见 :doc:`使能TensorCore示例代码 <../5_quickguide/quickguide>` 。需要注意的是，CUDA中矩阵乘和卷积计算是分开控制的，而在torch_musa中是统一控制的，代码差异如下所示。在torch_musa中 **allow_tf32** 默认值是False。

.. code-block:: python

  import torch
  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
  # in PyTorch 1.12 and later.
  torch.backends.cuda.matmul.allow_tf32 = True
  
  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
  torch.backends.cudnn.allow_tf32 = True

  import torch_musa
  # The flag below controls whether to allow TF32 on muDNN. This flag defaults to False.
  torch.backends.mudnn.allow_tf32 = True

在 **s4000** 上，对于具有卷积算子的模型，使能 **NHWC layout优化** ，可以使得性能得到提升，示例代码如下：

.. code-block:: python

  import torch
  import torch_musa

  torch.backends.mudnn.allow_tf32 = True
  model = Model() # define model here
  model = model.to(memory_format=torch.channels_last) # transform layout to NHWC
