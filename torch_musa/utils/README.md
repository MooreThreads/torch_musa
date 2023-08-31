# torch_musa utils
## MUSAExtension
MUSAExtension is a function which can do help to building MUSA backend for third party as CUDAExtension does the same thing for CUDA backend. What it differs from CUDAExtension is that it keeps the consistent interface with [CppExtension](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.CppExtension) so parameter `extra_compile_args` is a list instead of a dict. Unlike nvcc, it is not conveniently to integrate mcc (MUSA Compiler Collection) into MUSAExtension. But CmakeManager which is from pytorch/tools can be utilized to facilitate the mcc compiling part.
```
from torch_musa.utils.musa_extension import MUSAExtension

ext_module: torch.utils.cpp_extension.CppExtension = MUSAExtension(name, sources, *args, **kwargs)
```
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

Please refer to simple_porting.py if you want more customizations.

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
