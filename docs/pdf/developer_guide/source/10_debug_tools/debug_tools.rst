.. _comparetool:

异常算子对比和追踪工具使用指南
==============================

概述
--------

这个工具旨在通过提供跨设备比较张量算子、跟踪模块层次结构和检测NaN/Inf值的能力，增强PyTorch模型的调试和验证过程。它的目标是确保模型在开发和测试的各个阶段的正确性和稳定性。

功能
--------

基本用法
------------

**与CPU比较算子**

将张量算子的输出与CPU结果比较，以确保在不同设备上的一致性和正确性。这对于自定义算子或验证设备特定实现至关重要。

.. code-block:: python

    from torch_musa.utils.compare_tool import CompareWithCPU

    model = get_your_model()
    with CompareWithCPU(atol=0.001, rtol=0.001, verbose=True):
        train(model)

为了调试或性能评估，您可以使用``enabled``参数临时禁用比较：

.. code-block:: python

    with CompareWithCPU(enabled=False, atol=0.001, rtol=0.001, verbose=True):
        train(model)

**模块跟踪**

理解异常发生在哪个模块层次中与识别异常算子本身一样重要。要跟踪模块的层次结构并确定问题出现的位置，请启用模块跟踪器：

.. code-block:: python

    from torch_musa.utils.compare_tool import open_module_tracker, ModuleInfo

    model = get_your_model()
    open_module_tracker(model)
    with CompareWithCPU(atol=0.001, rtol=0.001, verbose=True):
        train(model)

**NaN/Inf检测**

虽然CompareWithCPU也能检测NaN和Inf值，但它需要在CPU上重新运行每个算子，这可能会很慢。为了快速识别NaN或Inf而不显著降低性能，可以使用``NanInfTracker``，它完全在GPU上运行：

.. code-block:: python

    from torch_musa.utils.compare_tool import NanInfTracker

    model = get_your_model()
    with NanInfTracker():
        train(model)

日志结构和解释
------------------------

每当模型执行算子时，该工具都会生成日志条目，包括算子的详细信息、输入输出数据、以及与CPU执行结果的比较。以下是一个典型的日志示例：

.. code-block:: bash

    2024-04-07, 15:11:10
    -------------------------  step = 1  ----------------------
    GeminiDDP/ChatGLMModel/torch.ops.aten.view(forward) is in white_list, pass
    GeminiDDP/ChatGLMModel/torch.ops.aten.ones(forward) starts to run ...
    GeminiDDP/ChatGLMModel/torch.ops.aten.ones(forward) succeeds to pass CompareWithCPU test
    ...

    GeminiDDP/ChatGLMModel/GLMTransformer/GLMBlock/SelfAttention/Linear/torch.ops.aten.addmm(forward) starts to run ...
    "addmm_impl_cpu_" not implemented for 'Half'
    Convert to float32 ...

    ============================
    [ERROR] GeminiDDP/ChatGLMModel/GLMTransformer/GLMBlock/SelfAttention/Linear/torch.ops.aten.addmm(forward) fails to pass CompareWithCPU test
    ....... input .........
    0: Tensor <shape=torch.Size([6144]), dtype=torch.float16, device=musa:0, size=6144, >, 
    1: Tensor <shape=torch.Size([24576, 5120]), dtype=torch.float16, device=musa:0, size=125829120, >, 
    2: Tensor <shape=torch.Size([5120, 6144]), dtype=torch.float16, device=musa:0, size=31457280, >, 


    ...... output ........
    Tensor <shape=torch.Size([24576, 6144]), dtype=torch.float16, device=musa:0, size=150994944, >

    ...... compare with cpu .......
    Tensor values are not close

    Too many indices (total 20581473) to print 

    ...

    Element at index (0, 14) is not close: -0.84521484375(musa:0) vs -0.8450137972831726(cpu)
    Element at index (0, 42) is not close: -1.1943359375(musa:0) vs -1.1947154998779297(cpu)
    Element at index (0, 46) is not close: -1.025390625(musa:0) vs -1.0250622034072876(cpu)
    Element at index (0, 52) is not close: 0.552734375(musa:0) vs 0.5529251098632812(cpu)
    Element at index (0, 54) is not close: 0.72216796875(musa:0) vs 0.7219759225845337(cpu)
    Element at index (0, 57) is not close: -1.310546875(musa:0) vs -1.3108956813812256(cpu)
    Element at index (0, 59) is not close: -0.52734375(musa:0) vs -0.5271496176719666(cpu)
    Element at index (0, 67) is not close: -0.5302734375(musa:0) vs -0.5304464101791382(cpu)
    Element at index (0, 77) is not close: 0.89306640625(musa:0) vs 0.8932651281356812(cpu)
    Element at index (0, 80) is not close: 0.56787109375(musa:0) vs 0.5681084394454956(cpu)
    Element at index (0, 84) is not close: 1.3388671875(musa:0) vs 1.338517427444458(cpu)
    Element at index (0, 95) is not close: -1.302734375(musa:0) vs -1.3023890256881714(cpu)
    Element at index (0, 100) is not close: -0.64306640625(musa:0) vs -0.6428374648094177(cpu)
    Element at index (0, 116) is not close: -0.79150390625(musa:0) vs -0.7917078733444214(cpu)
    Element at index (0, 130) is not close: -0.53271484375(musa:0) vs -0.5329336524009705(cpu)
    Element at index (0, 142) is not close: 1.2939453125(musa:0) vs 1.2935254573822021(cpu)
    Element at index (0, 146) is not close: -0.69970703125(musa:0) vs -0.6995066404342651(cpu)
    Element at index (0, 154) is not close: -0.5751953125(musa:0) vs -0.5753999352455139(cpu)
    Element at index (0, 156) is not close: 0.53759765625(musa:0) vs 0.5373584032058716(cpu)
    Element at index (0, 160) is not close: -0.56005859375(musa:0) vs -0.5602396726608276(cpu)

    ...

    Tensor <shape=torch.Size([24576, 6144]), dtype=torch.float16, device=musa:0, size=150994944, >
    tensor([[ 0.2247,  0.1085,  0.5469,  ...,  0.0325,  0.6895,  0.7295],
            [-0.7515, -0.6138, -0.5361,  ..., -0.7559,  1.2334,  0.7021],
            [ 0.0715,  0.1360, -1.0371,  ...,  0.6582,  0.8247, -0.0663],
            ...,
            [ 0.1399, -0.5474,  0.4290,  ...,  0.0474,  0.2852, -0.2908],
            [-0.5698, -0.1058, -0.5020,  ...,  0.2175, -0.4563, -0.5186],
            [ 0.6357, -0.9258, -0.2781,  ...,  0.8784, -0.5474, -0.0219]],
        device='musa:0', dtype=torch.float16)
    Tensor <shape=torch.Size([24576, 6144]), dtype=torch.float32, device=cpu, size=150994944, >
    tensor([[ 0.2247,  0.1085,  0.5469,  ...,  0.0325,  0.6893,  0.7297],
            [-0.7515, -0.6137, -0.5364,  ..., -0.7560,  1.2333,  0.7019],
            [ 0.0716,  0.1359, -1.0372,  ...,  0.6582,  0.8247, -0.0663],
            ...,
            [ 0.1399, -0.5473,  0.4289,  ...,  0.0474,  0.2851, -0.2909],
            [-0.5701, -0.1058, -0.5019,  ...,  0.2176, -0.4562, -0.5184],
            [ 0.6358, -0.9257, -0.2781,  ...,  0.8782, -0.5475, -0.0219]])

    ============================

* 算子细节：每个条目展示了执行的算子、其输入和输出细节，以及比较的结果。
* 错误识别：错误清晰标记，详细描述了设备间算子输出的差异。
* 定位异常：可以搜索 "[WARNING]" 来定位 NaN/Inf 出现的位置，搜索 "[ERROR]" 来定位与 CPU 比较失败的操作。

处理检测后的异常
-----------------------------

一旦检测到异常，该工具提供了几种策略来解决和解决这些问题：

1. **隔离特定算子**

   通过将有问题的算子添加到``target_list``，只比较异常算子，加快调试过程。

    .. code-block:: python

        from torch_musa.utils.compare_tool import CompareWithCPU, open_module_tracker

        model = get_your_model()
        open_module_tracker(model)
        with CompareWithCPU(atol=0.001, rtol=0.001, target_op=['torch.ops.aten.addmm']):
            train(model)

2. **调整公差**

    如果一个算子几乎通过比较，但刚好超出公差一点点，调整``atol``和``rtol``可能会有所帮助。

    .. code-block:: python

        from torch_musa.utils.compare_tool import CompareWithCPU, open_module_tracker

        model = get_your_model()
        open_module_tracker(model)
        with CompareWithCPU(atol=0.01, rtol=0.01, target_op=['torch.ops.aten.addmm']):
            train(model)

3. **白名单**

    对于已知和预期的异常算子，将算子添加到``white_list``可以将其从进一步比较中排除。

    .. code-block:: python

        from torch_musa.utils.compare_tool import CompareWithCPU, open_module_tracker

        model = get_your_model()
        open_module_tracker(model)
        with CompareWithCPU(atol=0.001, rtol=0.001, white_list=['torch.ops.aten.addmm']):
            train(model)

4. **调试和复现问题**

    对于不符合预期的异常，启用``dump_error_data``保存失败算子的输入/输出。程序会在第一次未通过比较测试时中断，异常算子的输入和输出分别保存在``path_to_save/op_name_inputs.pkl``和``path_to_save/op_name_outputs.pkl``中，方便单元测试复现。

    .. code-block:: python

        from torch_musa.utils.compare_tool import CompareWithCPU, open_module_tracker

        model = get_your_model()
        open_module_tracker(model)
        with CompareWithCPU(atol=0.01, rtol=0.01, verbose=True, target_op=['torch.ops.aten.addmm'], dump_error_data=True, output_dir='path_to_save'):
            train(model)
    
    然后用保存的输入和输出进行单元测试复现：

    .. code-block:: python

        from torch_musa.utils.compare_tool import compare_for_single_op

        correct, args, kwargs, out = compare_for_single_op('path_to_save/torch.ops.aten.addmm_inputs.pkl', torch.ops.aten.addmm, atol=0.01, rtol=0.01)

    只检测Nan/Inf时也类似：

    .. code-block:: python

        from torch_musa.utils.compare_tool import nan_inf_track_for_single_op

        correct, args, kwargs, out = nan_inf_track_for_single_op('path_to_save/torch.ops.aten.addmm_inputs.pkl', torch.ops.aten.addmm)

训练步骤控制
-------------

在AMP场景中，初始训练步骤的scale很大，容易产生NaN/Inf值，干扰异常算子的定位。通过设置``start_step``和``end_step``，并调用``step()``来增加``step_cnt``，控制何时激活比较或NaN/Inf跟踪。只有当``start_step <= step_cnt < end_step``时，CompareWithCPU和NanInfTracker才会激活。

.. code-block:: python

    from torch_musa.utils.compare_tool import open_module_tracker, ModuleInfo

    model = get_your_model()
    open_module_tracker(model)
    with CompareWithCPU(atol=0.001, rtol=0.001, verbose=True, start_step=5) as compare_with_cpu:
        for epoch in range(epoch_num):
            for step in range(step_num):
                train_step(model)
                compare_with_cpu.step()

分布式支持
----------------

``CompareWithCPU``和``NanInfTracker``本身就支持分布式设置。使用``should_log_to_file``开关避免日志中的跨rank干扰。

.. code-block:: python

    from torch_musa.utils.compare_tool import open_module_tracker, ModuleInfo

    model = get_your_model()
    open_module_tracker(model)
    with CompareWithCPU(atol=0.001, rtol=0.001, verbose=True, should_log_to_file=True, output_dir='path_to_save'):
        train(model)

此外，``enable_ranks``控制在特定rank上激活CompareWithCPU和NanInfTracker。这在一机多卡的场景下特别有用，可以避免所有rank使用同一块CPU进行算子比较。

.. code-block:: python

    from torch_musa.utils.compare_tool import open_module_tracker, ModuleInfo

    model = get_your_model()
    open_module_tracker(model)
    with CompareWithCPU(atol=0.001, rtol=0.001, verbose=True, should_log_to_file=True, output_dir='path_to_save', enable_ranks=[0]):
        train(model)

结论
--------

比较和跟踪工具是开发人员和研究人员在PyTorch模型上工作时的重要工具。它结合了几个高级功能，旨在增强模型开发过程，确保模型算子在各种计算环境中的准确性和可靠性。

- **与CPU比较算子**：此功能通过比较自定义或GPU特定实现的输出和标准CPU结果，实现了对张量算子的精确验证。确保算子的一致性和正确性至关重要，特别是在跨不同硬件平台部署模型时。

- **模块跟踪**：了解算子在模型中的上下文和层次结构可以显著简化调试和优化任务。模块跟踪允许开发人员将异常追溯到模型结构中的源头，提供数据如何通过网络流动的清晰理解。

- **NaN/Inf检测**：识别NaNs和Infs的出现对于诊断模型中的数值不稳定至关重要。NanInfTracker功能提供了一种快速有效的方法，直接在GPU上定位这些值，确保对性能的影响最小。

- **处理检测后的异常**：一旦检测到问题，该工具提供了几种隔离、分析和解决策略。这包括隔离特定算子进行集中比较，调整公差水平以适应微小差异，将预期的异常算子添加到白名单中，以及为意外的异常算子转储数据以促进详细的调试和单元测试复现。

- **训练步骤控制**：在诸如AMP训练之类的场景中，对何时激活比较和跟踪的自适应控制尤其有用，其中初始条件可能会产生误导性的NaN/Inf值。此功能允许有针对性的调试工作，在训练过程的最相关阶段再激活工具。

- **分布式支持**：该工具考虑到分布式计算来进行设计，支持无缝集成到分布式训练设置中。它提供了日志管理和在特定rank上选择性激活的功能，以优化多GPU环境中的性能和易用性。

通过提供一套全面的调试和验证工具，比较和跟踪工具显著促进了健壮、可靠和高性能PyTorch模型的开发。它对灵活性、效率和以用户为中心的设计的强调，使其成为任何旨在推动PyTorch可能性边界的开发人员工具包中的宝贵补充。
