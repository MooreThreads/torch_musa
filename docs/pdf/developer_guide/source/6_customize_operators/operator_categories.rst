算子实现分类
======================

实际上，PyTorch 算子就是 C++ 函数，不同后端提供对应的实现（包含引用第三方代码，比如 MUSA 的 .mu 文件），绑定到同一个接口；当接口被调用时，框架根据运行时环境和传入参数，推理出目标后端，然后派发到该后端注册的实现函数完成计算。

由于尺寸变化、类型转换或数据依赖等原因，算子的实现通常会先在内部创建临时变量，计算结果写入完成后作为结果返回，称为 functional 规则。以 isnan 算子为例，Python 接口为：

.. code-block:: python

  torch.isnan(input) → Tensor

输出是 bool 类型，当输入是 fp32 时，产生类型转换，结果变量只能从内部创建。

当实现逻辑不受上述因素影响时，算子可通过多种规则实现调用，表现为可选的输出参数。以 tril 算子为例，Python 接口为：

.. code-block:: python

  torch.tril(input, diagonal=0, *, out=None) → Tensor

可以看出，输出变量 out 是可配置的，支持外部传入。故产生了下列调用规则：

.. code-block:: python

  >>> a = torch.randn(3, 3)
  >>> torch.tril(a)

functional 规则（默认），计算结果由实现内部创建并返回。

.. code-block:: python

  >>> a = torch.randn(3, 3)
  >>> a.tril_()

inplace 规则，输入 a 是可读写的，计算完成后数据被覆盖。

.. code-block:: python

  >>> a = torch.randn(3, 3)
  >>> b = torch.randn(3, 3)
  >>> torch.tril(a, out=b)

out 规则，计算结果写入外部提前创建好的变量 b。

注意，PyTorch 算子区分调用规则。对于 tril 算子，native_functions.yaml 文件中对每种调用规则分别定义了 C++ 函数接口：

.. code-block:: c++

  Tensor tril(const Tensor&, int64_t);               // functional
  Tensor& tril_(Tensor&, int64_t);                   // inplace
  Tensor& tril_out(const Tensor&, int64_t, Tensor&); // out

Python 调用对应多种规则的 C++ 算子，称为一个 group；调用后，在 group 内选择对应的 C++ 算子进行派发。从实现角度看，不同的调用规则除了预处理有区别之外，计算逻辑是一致的。因此，PyTorch 针对 group 算子引入了 structured 的实现方式，整体逻辑拆分为预处理和计算两个子过程，这样的好处是：

- 预处理过程屏蔽调用规则的逻辑差异，对外不区分后端，甚至算子类型。
- 计算过程不区分调用规则，每个后端只需要实现一个计算函数。
- 算子实现的外层逻辑（预处理 + 计算）一致，可由 codegen 生成。

structured 的实现方式可以减少代码体积，实现高效的跨规则/跨算子逻辑复用，减少算子实现的工作量。与之相反，只有一种调用规则的算子，对应 unstructured 的实现方式（不存在多种调用规则的逻辑区别），开发者可以尝试复用已有的 structured 子过程，也可以独立实现整个算子逻辑。