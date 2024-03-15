profiler工具
-------------

可以使用PyTorch官方性能分析工具 **torch.profiler** 来对torch_musa进行性能分析，使用方法可以参考 `官方示例 <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_ 。需要注意的是，当前torch_musa不支持 **ProfilerActivity.MUSA** ，我们需要开启同步模式 ``export MUSA_LAUNCH_BLOCKING=1`` ,通过CPU耗时信息做性能分析。 **ProfilerActivity.MUSA** 相关特性正在开发中，在不久的新版本中torch_musa将会对其进行支持。


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
