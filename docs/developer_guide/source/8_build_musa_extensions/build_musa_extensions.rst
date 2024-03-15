本节主要介绍如何对PyTorch生态的第三方库进行MUSA扩展的构建(MUSAExtension)，对应于CUDAExtension。

为什么要对第三方库进行MUSA扩展的构建？
======================

以mmcv库(commit id为0a2f60ba0198f8d567b536313bfba329588f9c3f)为例，当我们的测试代码有如下报错log，则说明mmcv中没有构建MUSA扩展。此时，我们需要对mmcv库进行MUSA扩展，从而使得mmcv库运行在摩尔线程显卡上。

.. code-block:: python

  import numpy as np
  import torch
  import torch_musa
  from mmcv.ops import nms
  np_boxes = np.array([[6.0, 3.0, 8.0, 7.0], [3.0, 6.0, 9.0, 11.0],
                        [3.0, 7.0, 10.0, 12.0], [1.0, 4.0, 13.0, 7.0]],
                        dtype=np.float32)
  np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)
  np_inds = np.array([1, 0, 3])
  np_dets = np.array([[3.0, 6.0, 9.0, 11.0, 0.9],
                      [6.0, 3.0, 8.0, 7.0, 0.6],
                      [1.0, 4.0, 13.0, 7.0, 0.2]])
  boxes = torch.from_numpy(np_boxes)
  scores = torch.from_numpy(np_scores)
  # check if cpu can work
  dets, inds = nms(boxes, scores, iou_threshold=0.3, offset=0)
  # check if musa can work
  dets, inds = nms(boxes.musa(), scores.musa(), iou_threshold=0.3, offset=0)

.. figure:: ../doc_image/nms_impl_not_implemented.*

注意以上测试不要在mmcv根目录下进行，以免将当前目录下的mmcv包导入。

如何对第三方库进行MUSA扩展？
==============

了解MUSAExtension这个API
-------------
阅读torch_musa/utils/README.md中关于MUSAExtension的介绍。

CUDA-Porting
-------------

我们需要先找到与CUDA相关文件所在的位置，在mmcv中，有如下几处：

- mmcv/mmcv/ops/csrc/common/cuda/
- mmcv/mmcv/ops/csrc/pytorch/cuda/

为了方便我们将对mmcv/mmcv/ops/csrc这个目录进行CUDA-Porting，将会生成mmcv/mmcv/ops/csrc_musa目录。

同时也为了减少不必要的Porting，我们将如下几个目录进行忽略：

- mmcv/ops/csrc/common/mlu
- mmcv/ops/csrc/common/mps
- mmcv/mmcv/ops/csrc/parrots
- mmcv/mmcv/ops/csrc/pytorch/mlu
- mmcv/mmcv/ops/csrc/pytorch/mps
- mmcv/mmcv/ops/csrc/pytorch/npu

之后从mmcv根目录全局搜索cu、nv、cuda和对应的大写关键词。搜索关键词的目的在于梳理自定义的映射规则，本次对搜索结果的映射规则提取如下：

- _CU_H_ -> _MU_H_
- _CUH -> _MUH
- __NVCC__ -> __MUSACC__
- MMCV_WITH_CUDA -> MMCV_WITH_MUSA
- AT_DISPATCH_FLOATING_TYPES_AND_HALF -> AT_DISPATCH_FLOATING_TYPES
- #include <ATen/cuda/CUDAContext.h> -> #include \"torch_musa/csrc/aten/musa/MUSAContext.h\"
- #include <c10/cuda/CUDAGuard.h> -> #include \"torch_musa/csrc/core/MUSAGuard.h\"
- ::cuda:: -> ::musa::
- /cuda/ -> /musa/
- , CUDA, -> , PrivateUse1,
- .cuh -> .muh
- .is_cuda() -> .is_privateuseone()

大多数情况下，有一些基本的映射规则即cu->mu、nv->mt、cuda->musa、cuh->muh及对应的大写映射。如果在编译过程中遇到HALF相关的编译报错，
可以如上所示将HALF相关的宏取消掉。然后我们将搜索出来的关键词拓展，形成单词边界然后进行
映射，如果直接cu->mu那么就会产生Accumulate->Acmumulate这样的不期望的结果。第3、6、7、10、12个规则是一些固定的转换，其中
PrivateUse1是PyTorch中对于扩展的自定义backend默认名字，is_privateuseone也是属于自定义backend相关的API。

因此由上述分析我们得到如下CUDA-porting脚本：

.. code-block:: python
  
  SimplePorting(cuda_dir_path="./mmcv/ops/csrc", ignore_dir_paths=[
                "./mmcv/ops/csrc/common/mlu",
                "./mmcv/ops/csrc/common/mps",
                "./mmcv/ops/csrc/parrots",
                "./mmcv/ops/csrc/pytorch/mlu",
                "./mmcv/ops/csrc/pytorch/mps",
                "./mmcv/ops/csrc/pytorch/npu"
                ], 
                mapping_rule={
                    "_CU_H_": "_MU_H_",
                    "_CUH": "_MUH",
                    "__NVCC__": "__MUSACC__",
                    "MMCV_WITH_CUDA": "MMCV_WITH_MUSA",
                    "AT_DISPATCH_FLOATING_TYPES_AND_HALF": "AT_DISPATCH_FLOATING_TYPES",                  
                    "#include <ATen/cuda/CUDAContext.h>": "#include \"torch_musa/csrc/aten/musa/MUSAContext.h\"",
                    "#include <c10/cuda/CUDAGuard.h>": "#include \"torch_musa/csrc/core/MUSAGuard.h\"",
                    "::cuda::": "::musa::",
                    "/cuda/": "/musa/",
                    ", CUDA,": ", PrivateUse1,",
                    ".cuh": ".muh",
                    ".is_cuda()": ".is_privateuseone()",
                    }
                    ).run()

需要注意的是尽管我们自定义了映射规则，但是我们没有传入drop_default_mapping参数，因此在CUDA-porting时还会使用默认的映射规则，
见torch_musa/utils/mapping文件夹。由于文件夹下的general.json条目过多，并且基本上不会被用到，所以默认的映射规则里只包含除了
它之外的其他映射规则（mapping文件夹中除了general.json之外的其他json文件），general.json可作为自定义映射规则的参考。如果不
想在代码里添加映射规则，也可以在extra.json文件中添加条目或者自行添加新的json文件。

分析mmcv的构建脚本setup.py
-------------

.. code-block:: python
  
  ...  
  elif is_rocm_pytorch or torch.cuda.is_available() or os.getenv(
      'FORCE_CUDA', '0') == '1':
  if is_rocm_pytorch:
      define_macros += [('MMCV_WITH_HIP', None)]
  define_macros += [('MMCV_WITH_CUDA', None)]
  cuda_args = os.getenv('MMCV_CUDA_ARGS')
  extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
  op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
      glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp') + \
      glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cu') + \
      glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cpp')
  extension = CUDAExtension
  include_dirs.append(os.path.abspath('./mmcv/ops/csrc/pytorch'))
  include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))
  include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common/cuda'))
  elif (hasattr(torch, 'is_mlu_available') and  
  ...

在CUDA扩展的构建逻辑中，我们可以看到有环境变量'FORCE_CUDA'来控制是否构建，也可以看到有CUDA相关的宏定义'MMCV_WITH_CUDA'，赋值extension
为CUDAExtension，然后就是源文件以及头文件的设置。因此我们也可以加一个elif分支并利用环境变量'FORCE_MUSA'来控制是否构建，然后添加宏定义
'MMCV_WITH_MUSA'。为了方便，我们直接对mmcv/mmcv/ops/csrc这个目录进行CUDA-porting，会生成mmcv/mmcv/ops/csrc_musa。所以我们在设置源
文件以及头文件的路径时只需将csrc改为csrc_musa，最后将extension赋值为MUSAExtension就可以了。增加的分支如下所示：

.. code-block:: python

  ...  
  elif os.getenv('FORCE_MUSA', '0') == '1':
    from torch_musa.utils.simple_porting import SimplePorting
    from torch_musa.utils.musa_extension import MUSAExtension
    SimplePorting(cuda_dir_path="./mmcv/ops/csrc", ignore_dir_paths=[
    "./mmcv/ops/csrc/common/mlu",
    "./mmcv/ops/csrc/common/mps",
    "./mmcv/ops/csrc/parrots",
    "./mmcv/ops/csrc/pytorch/mlu",
    "./mmcv/ops/csrc/pytorch/mps",
    "./mmcv/ops/csrc/pytorch/npu"
    ], 
    mapping_rule={
        "_CU_H_": "_MU_H_",
        "_CUH": "_MUH",
        "__NVCC__": "__MUSACC__",
        "MMCV_WITH_CUDA": "MMCV_WITH_MUSA",
        "AT_DISPATCH_FLOATING_TYPES_AND_HALF": "AT_DISPATCH_FLOATING_TYPES",                  
        "#include <ATen/cuda/CUDAContext.h>": "#include \"torch_musa/csrc/aten/musa/MUSAContext.h\"",
        "#include <c10/cuda/CUDAGuard.h>": "#include \"torch_musa/csrc/core/MUSAGuard.h\"",
        "::cuda::": "::musa::",
        "/cuda/": "/musa/",
        ", CUDA,": ", PrivateUse1,",
        ".cuh": ".muh",
        ".is_cuda()": ".is_privateuseone()",
        }
        ).run()
    op_files = glob.glob('./mmcv/ops/csrc_musa/pytorch/*.cpp') + \
        glob.glob('./mmcv/ops/csrc_musa/pytorch/cpu/*.cpp') + \
        glob.glob('./mmcv/ops/csrc_musa/pytorch/cuda/*.mu') + \
        glob.glob('./mmcv/ops/csrc_musa/pytorch/cuda/*.cpp')
    define_macros += [
        ('MMCV_WITH_MUSA', None)
        ]
        
    extension = MUSAExtension
    include_dirs.append(os.path.abspath('./mmcv/ops/csrc_musa/pytorch'))
    include_dirs.append(os.path.abspath('./mmcv/ops/csrc_musa/common'))
    include_dirs.append(os.path.abspath('./mmcv/ops/csrc_musa/common/cuda'))
  elif (hasattr(torch, 'is_mlu_available') and  
  ...


由于构建MUSA扩展时生成了额外的共享库，然后它会被链接到最终的目标库，因此在安装包时它需要被一起打包，
具体示例见torch_musa/utils/README.md。mmcv中是以设置'include_package_data=True'和配置MANIFEST.in
文件来控制需要打包的数据的，原始MANIFEST.in文件如下所示：

.. code-block:: python

  include requirements/runtime.txt
  include mmcv/ops/csrc/common/cuda/*.cuh mmcv/ops/csrc/common/cuda/*.hpp mmcv/ops/csrc/common/*.hpp
  include mmcv/ops/csrc/pytorch/*.cpp mmcv/ops/csrc/pytorch/cuda/*.cu mmcv/ops/csrc/pytorch/cuda/*.cpp mmcv/ops/csrc/pytorch/cpu/*.cpp
  include mmcv/ops/csrc/parrots/*.h mmcv/ops/csrc/parrots/*.cpp
  include mmcv/ops/csrc/pytorch/mps/*.mm mmcv/ops/csrc/common/mps/*.h mmcv/ops/csrc/common/mps/*.mm
  recursive-include mmcv/ops/csrc/ *.h *.hpp *.cpp *.cuh *.cu *.mm

修改之后如下所示：

.. code-block:: python

  include requirements/runtime.txt
  include mmcv/ops/csrc_musa/common/cuda/*.cuh mmcv/ops/csrc_musa/common/cuda/*.hpp mmcv/ops/csrc_musa/common/*.hpp
  include mmcv/ops/csrc_musa/pytorch/*.cpp mmcv/ops/csrc_musa/pytorch/cuda/*.cu mmcv/ops/csrc_musa/pytorch/cuda/*.cpp mmcv/ops/csrc_musa/pytorch/cpu/*.cpp
  include mmcv/ops/csrc_musa/parrots/*.h mmcv/ops/csrc_musa/parrots/*.cpp
  include mmcv/ops/csrc_musa/pytorch/mps/*.mm mmcv/ops/csrc_musa/common/mps/*.h mmcv/ops/csrc_musa/common/mps/*.mm
  recursive-include mmcv/ops/csrc_musa/ *.h *.hpp *.cpp *.cuh *.cu *.mm

仅仅将csrc批量替换成csrc_musa即可。

尝试构建并测试
-------------
由于本次实验是在MTT S3000上进行，对应的MUSA架构版本为21，而mmcv中涉及到fp64的使用，所以我们也要打开这个选项。对于这些额外的环境变量，可以参
考torch_musa根目录下的CMakeLists.txt和build.sh。

接下来，我们尝试执行'MUSA_ARCH=21 ENABLE_COMPILE_FP64=1 FORCE_MUSA=1 python setup.py install > build.log'构建mmcv并记录构建日志。
很不幸，在第一次构建时遇到了一些编译错误，其中一个如下图所示：

.. figure:: ../doc_image/shared_memory_exceed_limit_error.*

这是由于定义的结构体（upfirdn2d_kernel_params）要使用的shared memory过大，超过了硬件（此次编译是在MTT S3000上进行的）规格的限制，因此我们尝试避免构建该kernel
的musa扩展（mmcv/mmcv/ops/csrc_musa/pytorch/cuda/upfirdn2d_kernel.mu）。如果您的模型中没有真实用到该kernel，那么可以将其注释起来，临时绕过该算子，保证模型的正常运行。
如果您的模型确认需要使用该kernel，那么请联系摩尔线程AI研发中心，反馈该问题（在外网提issue），我们及时修复。同理，对于其他的编译错误也是可以进行类似的修改。

汇总一下，我们对mmcv进行MUSA适配需要修改如下文件：

- MANIFEST.in
- mmcv/ops/csrc/common/cuda/carafe_cuda_kernel.cuh
- mmcv/ops/csrc/common/cuda/chamfer_distance_cuda_kernel.cuh
- mmcv/ops/csrc/common/cuda/scatter_points_cuda_kernel.cuh
- mmcv/ops/csrc/pytorch/cuda/upfirdn2d_kernel.cu
- setup.py

再次测试本节开头的例子，我们得到结果如下：

.. figure:: ../doc_image/case_passed.*

当然这并不能证明适配的mmcv的功能完全，我们可以对mmcv自带的单元测试进行简单的改动就可以进行测试了。如tests/test_ops/test_box_iou_quadri.py：

.. code-block:: python

  # Copyright (c) OpenMMLab. All rights reserved.
  import numpy as np
  import pytest
  import torch
  import torch_musa

  # from mmcv.utils import IS_CUDA_AVAILABLE


  class TestBoxIoUQuadri:

      @pytest.mark.parametrize('device', [
          'cpu',
          pytest.param(
              'musa',
              marks=pytest.mark.skipif(
                  not True, reason='requires MUSA support')),
      ])
      def test_box_iou_quadri_musa(self, device):
          from mmcv.ops import box_iou_quadri
          np_boxes1 = np.asarray([[1.0, 1.0, 3.0, 4.0, 4.0, 4.0, 4.0, 1.0],
                                  [2.0, 2.0, 3.0, 4.0, 4.0, 2.0, 3.0, 1.0],
                                  [7.0, 7.0, 8.0, 8.0, 9.0, 7.0, 8.0, 6.0]],
                                dtype=np.float32)
          np_boxes2 = np.asarray([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                                  [2.0, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 1.0],
                                  [7.0, 6.0, 7.0, 8.0, 9.0, 8.0, 9.0, 6.0]],
                                dtype=np.float32)
          np_expect_ious = np.asarray(
              [[0.0714, 1.0000, 0.0000], [0.0000, 0.5000, 0.0000],
              [0.0000, 0.0000, 0.5000]],
              dtype=np.float32)
          np_expect_ious_aligned = np.asarray([0.0714, 0.5000, 0.5000],
                                              dtype=np.float32)

          boxes1 = torch.from_numpy(np_boxes1).to(device)
          boxes2 = torch.from_numpy(np_boxes2).to(device)

          ious = box_iou_quadri(boxes1, boxes2)
          assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

          ious = box_iou_quadri(boxes1, boxes2, aligned=True)
          assert np.allclose(
              ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

      @pytest.mark.parametrize('device', [
          'cpu',
          pytest.param(
              'musa',
              marks=pytest.mark.skipif(
                  not True, reason='requires MUSA support')),
      ])
      def test_box_iou_quadri_iof_musa(self, device):
          from mmcv.ops import box_iou_quadri
          np_boxes1 = np.asarray([[1.0, 1.0, 3.0, 4.0, 4.0, 4.0, 4.0, 1.0],
                                  [2.0, 2.0, 3.0, 4.0, 4.0, 2.0, 3.0, 1.0],
                                  [7.0, 7.0, 8.0, 8.0, 9.0, 7.0, 8.0, 6.0]],
                                dtype=np.float32)
          np_boxes2 = np.asarray([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                                  [2.0, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 1.0],
                                  [7.0, 6.0, 7.0, 8.0, 9.0, 8.0, 9.0, 6.0]],
                                dtype=np.float32)
          np_expect_ious = np.asarray(
              [[0.1111, 1.0000, 0.0000], [0.0000, 1.0000, 0.0000],
              [0.0000, 0.0000, 1.0000]],
              dtype=np.float32)
          np_expect_ious_aligned = np.asarray([0.1111, 1.0000, 1.0000],
                                              dtype=np.float32)

          boxes1 = torch.from_numpy(np_boxes1).to(device)
          boxes2 = torch.from_numpy(np_boxes2).to(device)

          ious = box_iou_quadri(boxes1, boxes2, mode='iof')
          assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

          ious = box_iou_quadri(boxes1, boxes2, mode='iof', aligned=True)
          assert np.allclose(
              ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

我们进入到mmcv/tests/test_ops目录下，然后执行'pytest -s test_box_iou_quadri.py'就可以测试该单元测试用例了，测试结果如下所示：

.. figure:: ../doc_image/ut_passed.*


基于MUSAExtension构建定制化算子
======================

本小节以构建swintransformer中的 `window_process_kernel算子 <https://github.com/microsoft/Swin-Transformer/tree/main/kernels/window_process>`_ 为例介绍如何通过cuda-porting和MUSAExtension完成CUDA代码到MUSA代码的转译及构建。


原始目录结构如下：

:: 

    window_process
    ├── setup.py
    ├── swin_window_process.cpp
    ├── swin_window_process_kernel.cu
    ├── unit_test.py
    └── window_process.py

首先在window_process同级目录下执行cuda-porting命令将CUDA代码转化为MUSA代码：``python -m torch_musa.utils.simple_porting --cuda-dir-path ./window_process --mapping-rule "{\"cuda\":\"musa\",\"torch::kCUDA\":\"torch::kPrivateUse1\", \"type().is_cuda\":\"is_privateuseone\"}"``，命令执行完毕后会在当前目录下生成名为window_process_musa的新目录，该目录结构如下：

::

    window_process_musa
    ├── swin_window_process.cpp
    └── swin_window_process_kernel.mu

之后基于MUSAExtension编写setup.py文件，该文件写法与CUDAExtension类似，其代码如下：

.. code-block:: python

    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension
    from torch_musa.utils.musa_extension import MUSAExtension


    setup(
        name="swin_window_process",
        ext_modules=[
            MUSAExtension(
                "swin_window_process",
                [
                    "swin_window_process.cpp",
                    "swin_window_process_kernel.mu",
                ],
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )

在编译并安装MUSA扩展时需要通过MUSA_ARCH环境变量指定架构号，在S80或S3000机器上可执行 ``MUSA_ARCH=21 python setup.py install`` ,在S4000机器上可执行 ``MUSA_ARCH=22 python setup.py install`` 完成window_process_kernel扩展的安装，之后在python解释器中尝试 ``import swin_window_process`` 若无错误出现则表示定制化算子构建并安装成功。
