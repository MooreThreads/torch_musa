- [torch_musa utils](#torch_musa-utils)
  - [MUSAExtension](#musaextension)
    - [(Optional) MUSAExtension with half dtype in *.cpp files](#optional-musaextension-with-half-dtype-in-cpp-files)
  - [LOGGER](#logger)
  - [CMakeListsGenerator](#cmakelistsgenerator)
  - [SimplePorting](#simpleporting)
  - [Comparison and Tracking Tool](#comparison-and-tracking-tool)
    - [Overview](#overview)
    - [Features](#features)
      - [Basic Usage](#basic-usage)
        - [Operation Comparison with CPU](#operation-comparison-with-cpu)
        - [Module Tracking](#module-tracking)
        - [NaN/Inf Detection](#naninf-detection)
      - [Log Structure and Interpretation](#log-structure-and-interpretation)
      - [Handling Anomaly After Detection](#handling-anomaly-after-detection)
      - [Step Control](#step-control)
      - [Distributed Support](#distributed-support)
    - [Conclusion](#conclusion)
  - [musa-converter](#musa-converter)
# torch_musa utils

## MUSAExtension
MUSAExtension is a function that helps third-party libraries build MUSA extensions. It keeps the
consistent interface with CUDAExtension. We should take care of the different parts that the `extra_compile_args` introduces `mcc` key instead of `nvcc`,
and `cmdclass` should be passed by `BuildExtension` which is imported from `torch_musa.utils.musa_extension`

There is a simple MUSAExtension example which has the following catalogue structure:

![image](../../docs/images/render_dir_tree.png)

And the content of `setup_musa.py` is:
```
import os
from setuptools import setup, find_packages
from torch_musa.utils.simple_porting import SimplePorting
from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

c_flags = []
if os.name == "posix":
    c_flags = {
        "cxx": ['-O3', '-std=c++14'],
        "mcc": ["-O2"]   
    }

# porting .cu to .mu
SimplePorting(cuda_dir_path="freqencoder/src", mapping_rule={
    "x.device().is_cuda()": "true",
    "#include <ATen/cuda/CUDAContext.h>": "#include \"torch_musa/csrc/aten/musa/MUSAContext.h\"",
    "#include <c10/cuda/CUDAGuard.h>": "#include \"torch_musa/csrc/core/MUSAGuard.h\"",
    }).run()

setup(
    name='freqencoder', # package name, import this to use python API
    ext_modules=[
        MUSAExtension(
            name='freqencoder._MUSAC', # extension name, import this to use MUSA API
            sources=[os.path.join(_src_path, 'freqencoder/src_musa', f) for f in [
                'freqencoder.mu',
                'bindings.cpp',
            ]],
            extra_compile_args=c_flags
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### (Optional) MUSAExtension with half dtype in *.cpp files
If the `*.cpp` codes include `half` (aka float16) dtype, like these defined in `musa_fp16.h`, `cxx` compiler would throw an error that
`half` dtype is unrecognized. It's caused as the headers of `half` is different from CUDA's,
some macros are missded when applying `cxx` to compile these headers.

So it's kind different from `CUDAExtension` that we should introduce an argument
named `force_mcc` in `extra_compile_args`'s `cxx` entry, for example:
```python
c_flags = {
    "cxx": ["force_mcc", ...],
    "mcc": [...],
}
ext_modules=[
    MUSAExtension(
        name="xxx",
        sources=[...],
        extra_compile_args=c_flags,
    ),
    ...
]
```
and we would compile these cpp files with mcc compiler.

## LOGGER
```
from torch_musa.utils.logger_util import LOGGER

LOGGER.debug('debug')
LOGGER.info('info')
LOGGER.warning('warn')
LOGGER.error('error')
LOGGER.critical('critical')
```
## CMakeListsGenerator
```
from torch_musa.utils.cmake_lists_generator import CMakeListsGenerator

CMakeListsGenerator(sources=["/path/to/xxx.mu", "/path/to/xxx.cpp"], include_dirs=["/path/to/include_dir"], link_libraries="/path/to/libxxx.so"]).generate()
```
## SimplePorting

```
python -m torch_musa.utils.simple_porting --cuda-dir-path cuda/
```

`SimplePorting` outputs the transformed files to `${cuda-dir-path}_musa` hence executing the above command will generate a directory named `cuda_musa`. Please refer to simple_porting.py if you want more customizations.

Full command maybe:

```
python -m torch_musa.utils.simple_porting --cuda-dir-path cuda/ --ignore-dir-paths ["csrc/npu"] --mapping-rule {"cuda":"musa"} --drop-default-mapping --mapping-dir-path mapping/
```

If under WIN os then {"cuda":"musa"} should be '{\\"cuda\\":\\"musa\\"}'

If you want to integrate it to your own code then can use it like this:

```
from torch_musa.utils.simple_porting import SimplePorting

SimplePorting(cuda_dir_path, mapping_rule, drop_default_mapping, mapping_dir_path).run()
```

## Comparison and Tracking Tool

### Overview

This tool is designed to enhance the debugging and validation process of PyTorch models by offering capabilities for comparing tensor operations across devices, tracking module hierarchies, and detecting the presence of NaN/Inf values. It is aimed at ensuring the correctness and stability of models through various stages of development and testing.

### Features

#### Basic Usage

##### Operation Comparison with CPU

Compare tensor operation outputs against CPU results to ensure consistency and correctness across different devices. This is crucial for custom operations or verifying device-specific implementations.

```python
from torch_musa.utils.compare_tool import CompareWithCPU

model = get_your_model()
with CompareWithCPU(atol=0.001, rtol=0.001, verbose=True):
    train(model)
```

To temporarily disable the comparison for debugging or performance evaluation, you can use the `enabled` parameter:

```python
with CompareWithCPU(enabled=False, atol=0.001, rtol=0.001, verbose=True):
    train(model)
```

##### Module Tracking
Understanding where an anomaly occurs within the module hierarchy can be as crucial as identifying the anomaly itself. To track the hierarchy of modules and pinpoint where issues arise, enable the module tracker:

```python
from torch_musa.utils.compare_tool import open_module_tracker, ModuleInfo

model = get_your_model()
open_module_tracker(model)
with CompareWithCPU(atol=0.001, rtol=0.001, verbose=True):
    train(model)
```

##### NaN/Inf Detection

While CompareWithCPU inherently detects NaN and Inf values, it necessitates re-running each operation on the CPU, which may be slow. For rapid identification of NaN or Inf occurrences without significant performance degradation, the `NanInfTracker` operates entirely on the GPU:

```python
from torch_musa.utils.compare_tool import NanInfTracker

model = get_your_model()
with NanInfTracker():
    train(model)
```

#### Log Structure and Interpretation

The tool generates logs providing detailed insights into the operations being tested, including inputs, outputs, and comparison results:

```
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
```
- **Operation Details**: Each entry shows the operation being executed, its input and output details, and the comparison outcome.
- **Error Identification**: Errors are clearly marked, detailing discrepancies between the operation's outputs across devices.
- **Anomaly Localization**: You can search for "`[WARNING]`" to locate NaN/Inf occurrences and "`[ERROR]`" for operations failing the CPU comparison.


#### Handling Anomaly After Detection

Once an anomaly is detected, the tool offers several strategies to address and resolve these issues:

1. **Isolate Specific Operations**

   By adding the problematic operator to `target_list`, only the anomalous operation is compared, speeding up the debugging process.
   
    ```python
    from torch_musa.utils.compare_tool import CompareWithCPU, open_module_tracker

    model = get_your_model()
    open_module_tracker(model)
    with CompareWithCPU(atol=0.001, rtol=0.001, target_op=['torch.ops.aten.addmm']):
        train(model)
    ```

2. **Adjust Tolerances**

   If an operation almost passes the comparison but just falls outside the tolerance, adjusting `atol` and `rtol` might help.
   
    ```python
    from torch_musa.utils.compare_tool import CompareWithCPU, open_module_tracker

    model = get_your_model()
    open_module_tracker(model)
    with CompareWithCPU(atol=0.01, rtol=0.01, target_op=['torch.ops.aten.addmm']):
        train(model)
    ```

3. **Whitelisting**

   For anomalies that are known and expected, adding the operation to `white_list` can exclude it from further comparisons.
   
    ```python
    from torch_musa.utils.compare_tool import CompareWithCPU, open_module_tracker

    model = get_your_model()
    open_module_tracker(model)
    with CompareWithCPU(atol=0.001, rtol=0.001, white_list=['torch.ops.aten.addmm']):
        train(model)
    ```

4. **Debugging and Reproducing Issues**

   For unexpected anomalies, enabling `dump_error_data` saves inputs/outputs of failing operations. The program halts at the first failed comparison test, saving the anomalous operator's input and output in `path_to_save/op_name_inputs.pkl` and `path_to_save/op_name_outputs.pkl`, respectively, facilitating unit test reproduction.
   
    ```python
    from torch_musa.utils.compare_tool import CompareWithCPU, open_module_tracker

    model = get_your_model()
    open_module_tracker(model)
    with CompareWithCPU(atol=0.01, rtol=0.01, verbose=True, target_op=['torch.ops.aten.addmm'], dump_error_data=True,  output_dir='path_to_save'):
        train(model)
    ```

    Reproduce issues with the saved data:
   
    ```python
    from torch_musa.utils.compare_tool import compare_for_single_op

    correct, args, kwargs, out = compare_for_single_op('path_to_save/torch.ops.aten.addmm_inputs.pkl', torch.ops.aten.addmm, atol=0.01, rtol=0.01)
    ```

    Similarly, for NaN/Inf detection:
   
    ```python
    from torch_musa.utils.compare_tool import nan_inf_track_for_single_op

    correct, args, kwargs, out = nan_inf_track_for_single_op('path_to_save/torch.ops.aten.addmm_inputs.pkl', torch.ops.aten.addmm)
    ```

#### Step Control

In AMP scenarios, initial steps with large scales can produce NaN/Inf values, interfering with anomaly detection. Control when comparisons or NaN/Inf tracking activate by setting `start_step` and `end_step`, along with invoking `step()` to increment `step_cnt`. Comparisons activate only when `start_step <= step_cnt < end_step`.

```python
from torch_musa.utils.compare_tool import open_module_tracker, ModuleInfo

model = get_your_model()
open_module_tracker(model)
with CompareWithCPU(atol=0.001, rtol=0.001, verbose=True, start_step=5) as compare_with_cpu:
    for epoch in range(epoch_num):
        for step in range(step_num):
             train_step(model)
             compare_with_cpu.step()
```

#### Distributed Support
`CompareWithCPU` and `NanInfTracker` inherently support distributed setups. Use `should_log_to_file` to avoid cross-rank interference in logs.

```python
from torch_musa.utils.compare_tool import open_module_tracker, ModuleInfo

model = get_your_model()
open_module_tracker(model)
with CompareWithCPU(atol=0.001, rtol=0.001, verbose=True, should_log_to_file=True, output_dir='path_to_save'):
    train(model)
```

Additionally, `enable_ranks` controls the activation of CompareWithCPU and NanInfTracker on specific ranks, optimizing performance in multi-GPU setups by preventing all ranks from using the same CPU for operation comparison.

```python
from torch_musa.utils.compare_tool import open_module_tracker, ModuleInfo

model = get_your_model()
open_module_tracker(model)
with CompareWithCPU(atol=0.001, rtol=0.001, verbose=True, should_log_to_file=True, output_dir='path_to_save', enable_ranks=[0]):
    train(model)
```

### Conclusion

The Comparison and Tracking Tool is an essential asset for developers and researchers working on PyTorch models. It combines several advanced features designed to enhance the model development process, ensuring the accuracy and reliability of model operations across various computing environments.

- **Operation Comparison with CPU**: This feature enables precise validation of tensor operations by comparing outputs between custom or GPU-specific implementations and standard CPU results. It's crucial for ensuring the consistency and correctness of operations, especially when deploying models across different hardware platforms.

- **Module Tracking**: Understanding the context and hierarchy of operations within models can significantly simplify debugging and optimization tasks. Module tracking allows developers to trace anomalies back to their origins within the model structure, providing clear insights into how data flows through the network.

- **NaN/Inf Detection**: Identifying the occurrence of NaNs and Infs is vital for diagnosing numerical instabilities in models. The NanInfTracker feature offers a fast and efficient way to pinpoint these values directly on the GPU, ensuring minimal impact on performance.

- **Handling Issues After Detection**: Once an issue is detected, the tool offers several strategies for isolation, analysis, and resolution. This includes isolating specific operations for focused comparison, adjusting tolerance levels to accommodate minor discrepancies, whitelisting expected anomalies, and dumping data for unexpected anomalies to facilitate detailed debugging and unit test reproduction.

- **Step Control**: Adaptive control over when comparisons and tracking are activated can be particularly useful in scenarios like AMP training, where initial conditions might produce misleading NaN/Inf values. This feature allows for targeted debugging efforts, activating the tool's functionalities at the most relevant stages of the training process.

- **Distributed Support**: Designed with distributed computing in mind, the tool supports seamless integration into distributed training setups. It offers functionalities like logging management and selective activation on specific ranks to optimize performance and ease of use in multi-GPU environments.

By providing a comprehensive suite of debugging and validation tools, the Comparison and Tracking Tool significantly contributes to the development of robust, reliable, and high-performance PyTorch models. Its emphasis on flexibility, efficiency, and user-centric design makes it a valuable addition to the toolkit of any developer aiming to push the boundaries of what's possible with PyTorch.

## musa-converter
The musa-converter aimed for converting CUDA-related strings and APIs in scripts based on PyTorch to the MUSA platform and improve the efficiency of model migration from CUDA to MUSA.   
```shell
musa-converter -r ${/path/to/your/project} -l ${/path/to/your/project_launch_script}
```
Run `musa-converter -h` to see the detailed explanation of input parameters.