.. _comparetool:

CompareTool 使用指南
====================
如果发现模型准确率不对，可以通过算子精度对比工具来快速找出异常算子并进行修复。对比工具基于PyTorch的 `Dispatcher <http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/>`_ 和 `DispatchMode <https://dev-discuss.pytorch.org/t/torchdispatchmode-for-debugging-testing-and-more/717>`_
实现。在算子被dispatch到MUSA device之前捕获到调用函数，然后分别调用CPU算子函数和MUSA算子函数进行计算，得出结果进行对比。所有算子（包括前向算子和反向算子）都会被依次dispatch并输出对比结果到日志，通过查询日志可以快速定位出第一个出现计算异常的算子。该工具在训练和推理中都能被使用。

基本用法
--------

.. code-block:: python

    from torch_musa.utils.compare_tool import CompareWithCPU

    with CompareWithCPU(atol=0.001, rtol=0.001, verbose=True):
        ...original code...

在这段代码中，``CompareWithCPU`` 工具被用来比较在 MUSA 设备上执行算子作和在 CPU 上执行的相同算子的结果。这里， ``atol`` 和 ``rtol`` 分别代表绝对公差和相对公差，用于在比较时确定允许的误差范围。当 ``verbose`` 设置为 ``True`` 时，它会输出更详细的信息，包括输入输出tensor的shape、dtype、stride、nan_num等信息。

输出示例
--------

输出内容大致如下：

.. code-block:: none

    ============================
    ============================
    torch._ops.aten...div.Tensor
    ....... input .........
    0:  Tensor <shape=torch.Size([16, 3, 640, 640]), stride=(1228800, 409600, 640, 1), dtype=torch.float32, device=musa:0, size=19660800,>, 
    1: 255, 
    ...... output ........
    Tensor <shape=torch.Size([16, 3, 640, 640]), stride=(1228800, 409600, 640, 1), dtype=torch.float32, device=musa:0, size=19660800,>

    ...... compare with cpu .......
    torch._ops.aten...div.Tensor succeeds to pass CompareWithCPU test

    ============================
    torch._ops.aten..._to_copy.default
    ....... input .........
    0:  Tensor <shape=torch.Size([48, 3, 6, 6]), stride=(108, 1, 18, 3), dtype=torch.float32, device=musa:0, size=5184,>, 

    ...... output ........
    Tensor <shape=torch.Size([48, 3, 6, 6]), stride=(108, 1, 18, 3), dtype=torch.float16, device=musa:0, size=5184,>

    ...... compare with cpu .......
    torch._ops.aten..._to_copy.default succeeds to pass CompareWithCPU test


    ============================
    torch._ops.aten..._to_copy.default
    ....... input .........
    0:  Tensor <shape=torch.Size([16, 3, 640, 640]), stride=(1228800, 409600, 640, 1), dtype=torch.float32, device=musa:0, size=19660800,>, 

    ...... output ........
    Tensor <shape=torch.Size([16, 3, 640, 640]), stride=(1228800, 409600, 640, 1), dtype=torch.float16, device=musa:0, size=19660800,>

    ...... compare with cpu .......
    torch._ops.aten..._to_copy.default succeeds to pass CompareWithCPU test


    ============================
    torch._ops.aten...convolution.default
    ....... input .........
    0:  Tensor <shape=torch.Size([16, 3, 640, 640]), stride=(1228800, 409600, 640, 1), dtype=torch.float16, device=musa:0, size=19660800,>, 
    1:  Tensor <shape=torch.Size([48, 3, 6, 6]), stride=(108, 1, 18, 3), dtype=torch.float16, device=musa:0, size=5184,>, 
    2: None, 
    3: [2, 2, ], 
    4: [2, 2, ], 
    5: [1, 1, ], 
    6: False, 
    7: [0, 0, ], 
    8: 1, 

    ...... output ........
    Tensor <shape=torch.Size([16, 48, 320, 320]), stride=(4915200, 1, 15360, 48), dtype=torch.float16, device=musa:0, size=78643200,>

    ...... compare with cpu .......
    "slow_conv2d_cpu" not implemented for 'Half'
    Convert to float32 ...
    ........... output 0 is not close ........

    Too many indices (total 2388336) to print 
    ...
    Element at index (0, 3, 1, 75) is not close: 0.5458984375 vs 0.5460812449455261
    Element at index (0, 3, 1, 78) is not close: 0.5498046875 vs 0.5499854683876038
    Element at index (0, 3, 1, 86) is not close: 0.55029296875 vs 0.5501253008842468
    Element at index (0, 3, 1, 88) is not close: 0.55126953125 vs 0.5510903000831604
    Element at index (0, 3, 1, 91) is not close: 0.55029296875 vs 0.5504764914512634
    Element at index (0, 3, 1, 94) is not close: 0.54296875 vs 0.5427778959274292
    Element at index (0, 3, 1, 101) is not close: 0.5361328125 vs 0.5359049439430237
    Element at index (0, 3, 1, 103) is not close: 0.54638671875 vs 0.5466215014457703
    Element at index (0, 3, 1, 104) is not close: 0.54296875 vs 0.5431610941886902
    Element at index (0, 3, 1, 108) is not close: 0.54296875 vs 0.5427677631378174
    Element at index (0, 3, 1, 110) is not close: 0.5390625 vs 0.5392988920211792
    Element at index (0, 3, 1, 112) is not close: 0.5009765625 vs 0.5012078881263733
    Element at index (0, 3, 1, 114) is not close: 0.54052734375 vs 0.5403239130973816
    Element at index (0, 3, 1, 115) is not close: 0.5361328125 vs 0.5363231897354126
    Element at index (0, 3, 1, 117) is not close: 0.5234375 vs 0.5236586332321167
    Element at index (0, 3, 1, 118) is not close: 0.5029296875 vs 0.5027626156806946
    Element at index (0, 3, 1, 133) is not close: 0.537109375 vs 0.5373141765594482
    Element at index (0, 3, 1, 136) is not close: 0.513671875 vs 0.5134815573692322
    Element at index (0, 3, 1, 143) is not close: 0.5029296875 vs 0.5031570196151733
    Element at index (0, 3, 1, 144) is not close: 0.51953125 vs 0.519349217414856


    tensor 1: shape=torch.Size([16, 48, 320, 320]), numbers of nan = 0 of 78643200, numbers of inf = 0 of 78643200
    tensor([[[[-1.47583e-01, -1.81030e-01, -1.80420e-01,  ..., -1.57349e-01, -1.55396e-01, -1.71204e-02],
            [-1.83838e-01, -1.55762e-01, -1.54785e-01,  ..., -1.35498e-01, -1.33667e-01,  3.33252e-02],
            [-1.81885e-01, -1.56128e-01, -1.56738e-01,  ..., -1.50024e-01, -1.37939e-01,  3.04413e-02],
            ...,

            ..., ], device='musa:0', dtype=torch.float16)


    tensor 2 (golden): shape=torch.Size([16, 48, 320, 320]), numbers of nan = 0 of 78643200, numbers of inf = 0 of 78643200
    tensor([[[[-1.47546e-01, -1.81061e-01, -1.80469e-01,  ..., -1.57394e-01, -1.55356e-01, -1.71164e-02],
            [-1.83810e-01, -1.55728e-01, -1.54838e-01,  ..., -1.35546e-01, -1.33724e-01,  3.33278e-02],
            [-1.81920e-01, -1.56185e-01, -1.56751e-01,  ..., -1.49999e-01, -1.37939e-01,  3.04426e-02],
            ...,
            [-1.57577e-01, -1.27439e-01, -1.29771e-01,  ..., -1.27387e-01, -1.08608e-01,  1.68391e-02],
            [-1.57745e-01, -1.29170e-01, -1.31028e-01,  ..., -1.31670e-01, -1.23129e-01,  1.17836e-02],
            [ 1.78030e-02, -5.00134e-03, -7.76099e-03,  ..., -2.74925e-02, -2.82936e-02,  2.69481e-02]],
            ...
            ])
    all_resuls=[False]
    [ERROR] torch._ops.aten...convolution.default fails to pass CompareWithCPU test


这段输出显示了在 MUSA 设备上执行的 ``torch._ops.aten.div.Tensor``， ``torch._ops.aten._to_copy.default``， ``torch._ops.aten.convolution.default`` 算子和在 CPU 上执行的相同算子的输入、输出对比结果，以及它们是否成功通过了比较测试。

错误追踪与调试
--------------

如果在测试中发现错误或不一致，您可以在日志中搜索 "[WARNING]" 来追踪产生 nan/inf 的位置，搜索 "[ERROR]" 来追踪与 CPU 结果不一致的算子。然后，您可以调整 ``atol`` 和 ``rtol`` 的值再次尝试。或者设置 ``target_list`` 来快速重现异常算子行为，并设置 ``dump_error_data`` 来保存异常算子的输入数据：

.. code-block:: python

    from torch_musa.utils.compare_tool import CompareWithCPU

    with CompareWithCPU(atol=0.001, rtol=0.001, target_list=['convolution.default'], dump_error_data=True):
        ...original code...

这段代码将会把输入参数保存到 ``convolution.default_args.pkl`` 文件中，之后您可以用这个文件来生成单个错误用例进行调试：

.. code-block:: python

    import torch
    import torch_musa
    from torch_musa.utils.compare_tool import compare_single_op

    compare_single_op('convolution.default_args.pkl', torch.ops.aten.convolution.default, atol=0.0001, rtol=0.0001)

使用 ``compare_single_op`` 函数，您可以对特定的算子和输入进行更细致的比较和调试。


.. _DebugRegister:

调试注册接口使用指南：
=======================

目前，torch-musa中的部分算子已经使用调试注册接口进行了封装，可以在运行时通过环境变量启用或关闭调试功能。

使用说明：
-----------

在运行模型前，使用下列环境变量启用和配置调试功能：

TORCH_MUSA_OP_DEBUG=on            # 默认为关闭。指定为on时启用调试功能，off关闭。

TORCH_MUSA_OP_DEBUG_LEVEL=1       # 默认为1，可选1、2、3、4、5、6个等级。具体等级对应的功能见下文。注意：部分等级会严重影响性能，并可能占用大量磁盘空间。

TORCH_MUSA_OP_DEBUG_DIR=~/debug   # 指定调试日志的输出目录，默认为./DEBUG_DIR，推荐使用绝对路径。调试工具会自动创建该目录，如果目录已经存在，则会创建添加有数字编号后缀的新目录以避免重复。

TORCH_MUSA_OP_DEBUG_LIST=""       # 指定调试算子的名单，默认为未配置。使用英文逗号(,)分隔，不区分大小写，支持部分匹配。指定该环境变量时，只有算子名称或算子别名中包含列出的名称之一时，才会记录该算子的信息。当该环境变量未配置（unset）时，将默认包含所有算子。

TORCH_MUSA_OP_DEBUG_BLACK_LIST="" # 指定调试算子的黑名单，默认为未配置。使用英文逗号(,)分隔，不区分大小写，支持部分匹配。指定该环境变量时，只有算子名称和算子别名中不包含任意列出的名称时，才会记录该算子的信息。该环境变量不能和TORCH_MUSA_OP_DEBUG_LIST同时使用。

TORCH_MUSA_OP_DEBUG_LENGTH=50     # 指定调试张量数据类型时记录其部分值的最大长度N。默认为50。增加该数值将可能影响性能，并占用额外磁盘空间。


模式说明：
-----------

指定TORCH_MUSA_OP_DEBUG_LEVEL可以调整运行模式。目前支持6个不同等级的模式，使用1-6标识。

具体模式功能说明如下：（推荐使用1、2、3或5等级。4和6等级耗时极长并可能占用极大的磁盘空间！）

1：将所有调用到的算子，以及其数据规模、张量大小和标量值等信息记录到文件中。

2：在1的基础上，统计算子的张量信息，即记录其极大值、极小值、均值、方差等信息，并记录到文件中。（耗时较长）

3：在1的基础上，将算子张量的部分值（长度为N，可以使用TORCH_MUSA_OP_DEBUG_LENGTH进行配置）记录到文件中。

4：在1的基础上，将算子张量的全部值记录到文件中。（警告，该模式将可能记录大量数据，且耗时极长）。

5：2和3模式的组合，即统计张量信息的同时，也记录其部分值（长度为N）。（耗时较长）。

6：2和4模式的组合，即统计张量信息的同时，也记录其全部值。（警告，该模式将可能记录大量数据，且耗时极长）

使用本工具时，会在指定的目录下首先创建一个名为full_log.txt的文本文件，该文件记录了算子的基础信息。
同时，每个算子将会创建一个以其算子名称命名，并添加数字编号前缀的目录，目录中记录了算子内的数据信息。

使用调试注册接口可以在运行时记录模型中各个算子的数据，因此能够发现计算故障（如INF、NaN等）。
