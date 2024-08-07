算子开发
====================================

PyTorch 采用定义和实现分离的方式构建算子单元；定义部分包含算子格式、实现方式、后端绑定和导出规则（见 aten/src/ATen/native/native_functions.yaml 文件）；对于单个算子，官方为多种设备后端分别实现了计算逻辑。构建时，PyTorch 调用 torchgen 模块解析 yaml 文件，自动生成算子的接口（\*.h）和定义（\*.cpp）文件，后者包含了具体实现与后端的绑定，最终在运行时完成注册。

基于 yaml 文件的格式规范，torch_musa 扩展 torchgen 的部分逻辑实现了 codegen 模块；开发者实现 MUSA 算子的计算逻辑后，只需在 torch_musa/csrc/aten/ops/musa_functions.yaml 文件中添加该算子的关键描述，编译时 codegen 模块可自动解析文件内容，生成算子的接口和定义文件，完成与 MUSA 后端的绑定（参考 PyTorch 官方建议，torch_musa 复用 PrivateUse1 key 实现算子注册）。

本节旨在对 PyTorch 的算子实现进行分类，帮助开发者判断是否需要手动适配，选择合适的方式实现算子逻辑，修改 musa_functions.yaml 文件实现绑定注册，完成 MUSA 算子的开发。

.. note::
    对于单个算子，多后端实现共享相同的接口，在 musa_functions.yaml 文件中该算子的接口定义字段（- func:）只需要列出函数名即可。 

.. note::
    PyTorch 算子的 MUSA 后端 C++ 实现统一位于 at::musa 命名空间下，函数签名和接口定义一致，禁止默认参数值。

.. attention::
    本节不涉及算子正/反向计算关系的绑定；torch_musa 遵循 PyTorch 自动微分模块的设计与实现，绑定规则可见 tools/autograd/derivatives.yaml 文件。

.. toctree::
    operator_categories

.. toctree::
    when_need_to_customize_operators

.. toctree::
    customize_structured_operators

.. toctree::
    customize_unstructured_operators
