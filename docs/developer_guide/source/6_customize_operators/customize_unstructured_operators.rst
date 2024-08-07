实现 UnStructured 算子
====================================

PyTorch 的 unstructured 算子一般以 functional 规则调用，实现逻辑相互独立，不显式抽象出预处理和计算子逻辑，而是在实现内部自组织。算子定义时，我们需要显式指定 MUSA 后端的派发名，和 structured 算子不同，这个名字就是实现绑定的函数名。以 nonzero 算子为例，参考 CPU/CUDA 声明格式，MUSA 后端可声明如下：

.. code-block:: yaml

  # CPU/CUDA 声明，来自 native_functions.yaml
  # structured 默认为 false，标记为 unstructured 实现方式
  - func: nonzero(Tensor self) -> Tensor
    dispatch:
      CPU: nonzero_cpu
      CUDA: nonzero_cuda

  # MUSA 声明，来自 musa_functions.yaml
  # structured 缺省，默认和官方保持一致
  - func: nonzero
    dispatch:
      PrivateUse1: Nonzero

算子声明可以显式指定 structured 为 false，由于官方实现方式默认是 unstructured，这里也可以忽略不写，codegen 模块自动对齐 CPU/CUDA 的设置。考虑灵活扩展，torch_musa 也支持官方 structured 的实现转换为 MUSA 后端的 unstructured 实现，以 add.Tensor 算子为例（Tensor + Tensor），MUSA 的 unstructured 声明如下：

.. code-block:: yaml

  # Unstructured 声明，来自 musa_functions.yaml
  - func: add.Tensor
    dispatch:
      PrivateUse1: AddTensor
  - func: add_.Tensor
    dispatch:
      PrivateUse1: AddTensor_
  - func: add.out
    structured: false
    dispatch:
      PrivateUse1: AddTensorOut

如果算子的原始 structured 声明包含下列属性，在 MUSA 声明中需要显式处理：

- structured：out 声明用 false 值覆盖，functional/inplace 声明缺省。
- structured_delegate：functional/inplace 声明用 none 覆盖，out 声明缺省。
- structured_inherits：out 声明用 none 值覆盖，functional/inplace 声明缺省。
- precomputed：out 声明用 none 值覆盖，functional/inplace 声明缺省。
- dispatch：所有规则声明都需要显式指定派发的后端名。

正确设置上述属性后，codegen 模块把 group 内每个调用规则的算子看成是实现独立的，生成注册函数时内部直接调用 MUSA 后端绑定名对应的实现函数，否则解析 yaml 文件会报错。在算子实现部分，我们可以根据实际情况选择最合适的逻辑策略。

复用公共函数（Legacy）
------------------

当算子的实现逻辑完全不区分后端时，CPU/CUDA 的实现会共用 PyTorch 仓库提供的一个基础函数，内部逻辑可能只涉及非数据计算的视图转换，或者数据计算完全由依次调用其他算子完成。以 view_as_real 算子为例，实现逻辑只是复数拆开成两个浮点数的视图转换，没有数据计算，原始的官方声明如下：

.. code-block:: yaml

  - func: view_as_real(Tensor(a) self) -> Tensor(a)
    dispatch:
      CPU, CUDA, MPS, Meta: view_as_real

可以看出 CPU/CUDA 等后端绑定了一个公共的 view_as_real 函数实现转换；MUSA 后端实现应该保持一致，避免冗余代码，表现在算子声明上：

.. code-block:: yaml

  - func: view_as_real
    dispatch:
      PrivateUse1: view_as_real

在编译时，codegen 模块会在 unstructured 实现方式的前提下对比函数名，判断出 MUSA 的绑定函数和 CPU/CUDA 一样，注册函数内部自动调用对应的公共函数，也称为 Legacy 实现。

接入 MUDNN（Customized）
------------------

如果算子的 MUSA 实现涉及数据计算，正好 MUDNN 库提供了相应的能力时，我们可以在实现逻辑内部直接调用 MUDNN 接口完成计算。使用 MUDNN 库的主要步骤如下：

1. 参数校验，主要检查 device/dtype/defined_tensor 等。
2. 添加 DeviceGuard。
3. 参数转换，比如 conv 算子只支持连续 tensors，需要提前把输入/输出 tensors 转换为连续的。 
4. 创建输入/输出 MUTensors，以及 MUDNN 的计算实例，配置计算参数。
5. 调用实例接口完成计算，返回计算结果。

以 add.Tensor 的 functional 算子为例，我们可以在 torch_musa/csrc/aten/ops 目录下创建 Add.cpp 文件，实现 AddTensor 函数：

.. code-block:: c++

  #include <mudnn.h>
  
  Tensor AddTensor(
      const Tensor& self, const Tensor& other, Scalar const& alpha_scalar) {
    // 检查 device
    TORCH_CHECK(self.device().type() == kMUSA, "......");
    TORCH_CHECK(self.device().type() == other.device().type(), "......");
    
    // 检查 dtype
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float, "......");
    
    ......

    // 添加 DeviceGuard
    const c10::musa::MUSAGuard guard(self.device());

    // 连续性转换
    auto self_contig = self.contiguous();
    auto other_contig = at::mul(other, alpha_scalar);
    other_contig = other_contig.contiguous();

    // 创建输出
    auto output = at::empty(
      infer_size_dimvector(self.sizes(), other.sizes()),
      self.options());

    // 创建 MUTensors
    muTensor lhs = CreateMUTensor(self_contig);
    muTensor rhs = CreateMUTensor(other_contig);
    muTensor out = CreateMUTensor(output);

    // 调用 MUDNN 接口
    auto& h = GetMudnnHandle();
    ::musa::dnn::Binary op;
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Binary::Mode::ADD), "SetMode");
    CHECK_MUDNN_STATUS(op.Run(h, out, lhs, rhs), "Run Add.Tensor");

    return output;
  }

通过 mudnn*.h 头文件我们可以查询 MUDNN 库的算子支持情况和接口定义，默认地址为 /usr/local/musa/include 目录。如果 MUDNN 不支持，我们也可以通过 CUDA-Porting 等方式手动实现 MUSA kernels，在函数内部手工调用完成计算。

CPU 计算（Customized）
------------------

对于部分算子，如果MUDNN不支持，CUDA-Porting也无法支持，可以临时中转到 CPU 后端实现该算子。主要逻辑是，先把 tensor拷贝到 CPU 上，调用 CPU 算子完成计算，再将结果拷贝回 GPU。可以参考下述代码：

.. code-block:: c++

    Tensor AddTensor(
        const Tensor& self, const Tensor& other, Scalar const& alpha_scalar) {
      const auto cpu_dev = DeviceType::CPU;
      const auto musa_dev = self.device();
      auto cpu_self = at::empty(self.sizes(), self.options().device(cpu_dev));
      auto cpu_other = at::empty(other.sizes(), other.options().device(cpu_dev));
      return at::cpu::add(cpu_self, cpu_other).to(musa_dev);
    }

在初次适配模型时，可以通过这种方式快速判断缺少哪些算子，然后再逐个适配，通过接入 MUDNN 或自定义 kernels 的方式提高性能。

.. attention::
    除了 MUSA 后端（PrivateUse1）外，其他 MUSA 相关后端（比如 QuantizedPrivateUse1）的实现方式一定是 unstructured 的。

.. note::
    以上代码仅作参考，不代表实际的实现逻辑。