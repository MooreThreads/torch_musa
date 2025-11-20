是否需要适配算子
======================

以 tril 算子为例，我们可以尝试执行下面的 Python 程序：

.. code-block:: python

  import torch
  import torch_musa
  input_data = torch.randn(3, 3, device="musa")
  result = torch.tril(input_data)

当 torch_musa 内部实现了该算子时，程序正常执行完毕，可以查看 result 的数据和属性：

.. figure:: ../doc_image/tril_implemented.*

当测试环境打印如下报错信息:

.. figure:: ../doc_image/tril_not_implemented.*

可以确定，MUSA 后端缺少该算子的实现，原因为:

1. 尝试调用 functional 规则的算子，即内部创建 result ，计算结果写入后返回，发现该实现缺失。

2. 先创建 result，作为 out 参数传入 out 规则的算子调用，发现该实现也缺失。

3. 没有其他可调用的算子，报错返回异常。

此时需要我们手动实现 tril 算子。考虑计算逻辑的完备性，建议把 tril group 内所有调用规则对应的 C++ 算子全都实现。