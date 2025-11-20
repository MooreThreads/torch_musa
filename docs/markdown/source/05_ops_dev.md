---
title: 算子开发
description: torch_musa 算子开发
hide_table_of_contents: False
---

# 算子开发

PyTorch 采用定义和实现分离的方式构建算子单元；定义部分包含算子格式、实现方式、后端绑定和导出规则（见 aten/src/ATen/native/native_functions.yaml 文件）；对于单个算子，官方为多种设备后端分别实现了计算逻辑。构建时，PyTorch 调用 torchgen 模块解析 yaml 文件，自动生成算子的接口（\*.h）和定义（\*.cpp）文件，后者包含了具体实现与后端的绑定，最终在运行时完成注册。

基于 yaml 文件的格式规范，torch_musa 扩展 torchgen 的部分逻辑实现了 codegen 模块；开发者实现 MUSA 算子的计算逻辑后，只需在 torch_musa/csrc/aten/ops/musa_functions.yaml 文件中添加该算子的关键描述，编译时 codegen 模块可自动解析文件内容，生成算子的接口和定义文件，完成与 MUSA 后端的绑定（参考 PyTorch 官方建议，torch_musa 复用 PrivateUse1 key 实现算子注册）。

本节旨在对 PyTorch 的算子实现进行分类，帮助开发者判断是否需要手动适配，选择合适的方式实现算子逻辑，修改 musa_functions.yaml 文件实现绑定注册，完成 MUSA 算子的开发。

:::info 注解：

对于单个算子，多后端实现共享相同的接口，在 musa_functions.yaml 文件中该算子的接口定义字段（- func:）只需要列出函数名即可。

:::

:::info 注解：
PyTorch 算子的 MUSA 后端 C++ 实现统一位于 at::musa 命名空间下，函数签名和接口定义一致，禁止默认参数值。

:::

:::warning 注意：
本节不涉及算子正/反向计算关系的绑定；torch_musa 遵循 PyTorch 自动微分模块的设计与实现，绑定规则可见 tools/autograd/derivatives.yaml 文件。

:::

## 算子实现分类

实际上，PyTorch 算子就是 C++ 函数，不同后端提供对应的实现（包含引用第三方代码，比如 MUSA 的 .mu 文件），绑定到同一个接口；当接口被调用时，框架根据运行时环境和传入参数，推理出目标后端，然后派发到该后端注册的实现函数完成计算。

由于尺寸变化、类型转换或数据依赖等原因，算子的实现通常会先在内部创建临时变量，计算结果写入完成后作为结果返回，称为 functional 规则。以 isnan 算子为例，Python 接口为：

```python
torch.isnan(input) → Tensor
```

输出是 bool 类型，当输入是 fp32 时，产生类型转换，结果变量只能从内部创建。

当实现逻辑不受上述因素影响时，算子可通过多种规则实现调用，表现为可选的输出参数。以 tril 算子为例，Python 接口为：

```python
torch.tril(input, diagonal=0, *, out=None) → Tensor
```

可以看出，输出变量 out 是可配置的，支持外部传入。故产生了下列调用规则：

```python
>>> a = torch.randn(3, 3)
>>> torch.tril(a)
```

functional 规则（默认），计算结果由实现内部创建并返回。

```python
>>> a = torch.randn(3, 3)
>>> a.tril_()
```

inplace 规则，输入 a 是可读写的，计算完成后数据被覆盖。

```python
>>> a = torch.randn(3, 3)
>>> b = torch.randn(3, 3)
>>> torch.tril(a, out=b)
```

out 规则，计算结果写入外部提前创建好的变量 b。

注意，PyTorch 算子区分调用规则。对于 tril 算子，native_functions.yaml 文件中对每种调用规则分别定义了 C++ 函数接口：

```c
Tensor tril(const Tensor&, int64_t);               // functional
Tensor& tril_(Tensor&, int64_t);                   // inplace
Tensor& tril_out(const Tensor&, int64_t, Tensor&); // out
```

Python 调用对应多种规则的 C++ 算子，称为一个 group；调用后，在 group 内选择对应的 C++ 算子进行派发。从实现角度看，不同的调用规则除了预处理有区别之外，计算逻辑是一致的。因此，PyTorch 针对 group 算子引入了 structured 的实现方式，整体逻辑拆分为预处理和计算两个子过程，这样的好处是：

- 预处理过程屏蔽调用规则的逻辑差异，对外不区分后端，甚至算子类型。
- 计算过程不区分调用规则，每个后端只需要实现一个计算函数。
- 算子实现的外层逻辑（预处理 + 计算）一致，可由 codegen 生成。

structured 的实现方式可以减少代码体积，实现高效的跨规则/跨算子逻辑复用，减少算子实现的工作量。与之相反，只有一种调用规则的算子，对应 unstructured 的实现方式（不存在多种调用规则的逻辑区别），开发者可以尝试复用已有的 structured 子过程，也可以独立实现整个算子逻辑。

## 是否需要适配算子

以 tril 算子为例，我们可以尝试执行下面的 Python 程序：

```python
import torch
import torch_musa
input_data = torch.randn(3, 3, device="musa")
result = torch.tril(input_data)
```

当 torch_musa 内部实现了该算子时，程序正常执行完毕，可以查看 result 的数据和属性：

![tril_implemented.png](https://github.com/MooreThreads/torch_musa/blob/v1.3.0/docs/developer_guide/source/doc_image/tril_implemented.png?raw=true)

当测试环境打印如下报错信息:

![tril_not_implemented.png](https://github.com/MooreThreads/torch_musa/blob/v1.3.0/docs/developer_guide/source/doc_image/tril_not_implemented.png?raw=true)

可以确定，MUSA 后端缺少该算子的实现，原因为:

1. 尝试调用 functional 规则的算子，即内部创建 result ，计算结果写入后返回，发现该实现缺失。
2. 先创建 result，作为 out 参数传入 out 规则的算子调用，发现该实现也缺失。
3. 没有其他可调用的算子，报错返回异常。

此时需要我们手动实现 tril 算子。考虑计算逻辑的完备性，建议把 tril group 内所有调用规则对应的 C++ 算子全都实现。

## 实现Structured算子

对于 structured group 内多种调用规则的 C++ 算子，PyTorch 使用多级继承的方式实现统一的预处理和计算逻辑，继承类的命名遵循 torchgen 的规则，添加 “structured” 前缀。以 tril 算子的 CUDA 实现为例，规则如下（由底至上）：

1. 以 functional 算子名创建 meta::structured_tril 类（简称 meta 类，继承官方的 MetaBase 基类），新增 meta 函数实现预处理。
2. 以 out 算子绑定的后端实现名创建 native::structured_tril_cuda 类（简称 impl 类，继承 meta 类），新增 impl 函数实现计算。
3. 创建 structured_tril_cuda_functional 类（简称 functional 类，继承 impl 类），覆写基类方法实现结果 tensor 的创建。
4. 创建 structured_tril_cuda_inplace 类（简称 inplace 类，继承 impl 类），覆写基类方法实现可读写 tensor 的校验。
5. 创建 structured_tril_cuda_out 类（简称 out 类，继承 impl 类），覆写基类方法实现结果 tensor 的校验和尺寸变化。
6. functional/inplace/out 算子实现为分别实例化 3/4/5 中创建的子类，依次调用 meta 和 impl 函数，最后返回结果。

作为开发者，算子的实现部分我们只需完成 meta 类的预处理和 impl 类的计算逻辑开发，实现对应的函数；为了确保 codegen 正确生成算子文件和注册绑定，native_functions.yaml 文件中对三种调用规则的算子定义添加如下字段：

- 不同调用规则的算子类（上述 3/4/5 中的类）继承同一个 impl 类，视为一种 “规约” 关系；impl 类名固定引用 out 算子绑定的后端名，故在 functional/inplace 算子定义中添加 “规约” 字段，目标为 out 算子名。

```yaml
- func: tril(Tensor self, int diagonal=0) -> Tensor
  # 参考 out 算子的内容，生成 functional 类
  # 标记该算子为 structured 的实现方式
  structured_delegate: tril.out

- func: tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
  # 参考 out 算子的内容，生成 inplace 类
  # 标记该算子为 structured 的实现方式
  structured_delegate: tril.out
```

- out 算子声明除了后端名外，需要增加 structured 实现方式的标记，支持扩展字段描述基类信息，以及传递预处理的中间变量映射关系。

```yaml
- func: tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
  structured: True  # 标记该算子为 structured 的实现方式
  dispatch:    CUDA: tril_cuda # out 类和 impl 类实现的命名参考
  # 选择要继承的基类，缺省默认为 MetaBase，自定义例如
  # structured_inherits: TensorIteratorBase
  # 预处理传递给计算函数的中间变量映射关系，缺省默认为原参数传递，自定义例如
  # precomputed:
  # - diagonal -> int var
```

导入上述定义字段后，构建时 torchgen 自动解析字段内容，生成上述 1-6 中所有的类定义和算子实现（调用 meta 和 impl 函数），生成绑定和注册代码，完成算子的适配。

基于上述规范，在实现 structured 的 MUSA 算子时，考虑到 PyTorch 已经集成了 CUDA 的实现，meta/impl 函数可能都可以复用，故有三种实现方法。

### 复用meta/impl函数（Legacy）

这类算子的 meta 预处理逻辑是通用的，impl 函数内部应用了 PyTorch 的 DispatchStub 机制，即 CPU/CUDA 后端分别把计算 kernel 注册到算子对应的 stub 中，运行时根据目标 device 在 stub 中找到对应的 kernel 完成计算。我们可以直接复用 CUDA 的 impl 实现，只需要实现 MUSA kernel, 并注册到对应的 stub 中即可。

!!! note 注解：
    DispatchStub 的原理可参考 pytorch/aten/src/ATen/native/DispatchStub.h 源码文件

由于 MUSA 的编程模型实现了 CUDA 兼容，MUSA kernel 的一种实现方法是通过 CUDA-Porting 工具完成，主要流程如下：

1. 新建目录 build/generated_cuda_compatible，保存 Porting 的 kernels 文件和依赖头文件。
2. 把 PyTorch 仓库中的 CUDA kernels 和安装目录中的头文件复制到新建目录内（维持相对路径）。
3. Porting 工具实施文本替换，如将 cudaMalloc 替换成 musaMalloc，cuda_fp16.h 替换成 musa_fp16.h 等。

上述步骤在编译时依次执行，结束后指定目录下出现目标 mu 文件，包含转换完成的 MUSA kernel 和 stub 注册实现，我们把该文件添加到 musa_kernels 库包含的源文件集合中即可。以 lerp 算子为例，目标文件为 Lerp.mu，包含如下内容：

```cpp
// 文件位置：build/generated_cuda_compatible/aten/src/ATen/native/musa/Lerp.mu

// lerp.Tensor group MUSA kernel
void lerp_tensor_kernel(at::TensorIteratorBase& iter) {......}

// lerp.Scalar group MUSA kernel
void lerp_scalar_kernel(at::TensorIteratorBase& iter, const c10::Scalar& weight) {
  ......
}

// lerp.Tensor group stub 注册
REGISTER_DISPATCH(lerp_kernel_tensor_weight, &lerp_tensor_kernel);
// lerp.Scalar group stub 注册
REGISTER_DISPATCH(lerp_kernel_scalar_weight, &lerp_scalar_kernel);
```

可以看到文件中已经包含了计算和注册代码，我们在编译文件中做如下修改：

```yaml
# 文件位置：/home/torch_musa/torch_musa/csrc/CMakeLists.txt

file(
  GLOB_RECURSE
  MU_SRCS
  ......
  ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Lerp.mu
  ......
)
```

编译完成后 musa_kernels 动态库中会包含 MUSA lerp 的 kernels，在运行时自动注册到 lerp stub 中。 由于 MUSA 和 CUDA 设备的参数和架构存在差异，有时通过 CUDA-Porting 生成的 MUSA kernels 在运行时会报错或者效率不高，此时开发者可以选择手动修改或重写计算逻辑，即放弃在编译文件中加入 Porting 生成的 Lerp.mu 文件，在 torch_musa/csrc/aten/ops/musa 目录下创建 Lerp.mu 文件，手动实现 lerp kernels 和 stub 注册。

对于这种 Legacy 的实现方式，musa_functions.yaml 中算子的定义只需要列出算子名，其他的 structured 关键字 codegen 模块自动与 PyTorch 保持对齐。以 Lerp 算子为例，内容如下：

```yaml
- func: lerp.Scalar_out
- func: lerp.Scalar
- func: lerp_.Scalar

- func: lerp.Tensor_out
- func: lerp.Tensor
- func: lerp_.Tensor
```

### 只复用meta函数（LegacyMeta)

当算子的预处理逻辑通用，计算逻辑不使用 DispatchStub 时，开发者需要显式实现 MUSA 后端的 impl 函数。以 tril 算子为例，我们首先在 musa_functions.yaml 文件中添加算子接口定义：

```yaml
- func: tril
- func: tril_
- func: tril.out
  dispatch:    PrivateUse1: MusaTril
```

functional/inplace 规则的算子 “规约” 方式保持一致，列出算子名即可；与 Legacy 方式不同，out 算子需要显式指定后端实现名，让 codegen 模块自动生成 impl 类（在 at::musa 命名空间下）。实现计算逻辑时，我们可以在 torch_musa/csrc/aten/ops 目录下新建 Tril.cpp 文件，实现如下函数：

```c
// 文件位置：torch_musa/csrc/aten/ops/Tril.cpp
namespace at::musa {

TORCH_IMPL_FUNC(MusaTril)(const Tensor& self, int64_t k, const Tensor &result) {  // 计算过程
}} // namespace at::musa
```

PyTorch 针对 structured 类和函数定义了一系列宏，此处 TORCH_IMPL_FUNC(MusaTril) 会自动展开为 void structured_MusaTril::impl，与 codegen 生成的 impl 类函数保持一致。LegacyMeta 的实现方式不用修改编译文件，Tril.cpp 会自动被加入到 musa_kernels 库的源文件集合中。

### 自定义meta/impl函数（Customized）

如果在实现MUSA structured算子时遇到如下情况：

1. 预处理逻辑和 CPU/CUDA 有区别。
2. meta 类需要继承不同的基类。
3. meta 函数需要传给 impl 函数自定义的中间值，与 CPU/CUDA 不同或 CPU/CUDA 不传中间值。

开发者需要同时显式实现 meta 和 impl 函数（都在 at::musa 命名空间下）。以 tril 算子为例，我们首先在 musa_functions.yaml 文件中添加算子接口定义：

```yaml
- func: tril
- func: tril_
- func: tril.out
  structured_inherits: MyMetaBase  # 集成的基类名
  precomputed:  - diagonal -> int var # 需要传递的中间变量
  dispatch:    PrivateUse1: MusaTril
```

out 算子定义中必须显式指定 structured_inherits（情况2）或者 precomputed （情况3）字段。考虑情况 2 ，meta 类的定义如下：

```c
namespace at::musa {

struct TORCH_API structured_tril : public at::musa::MyMetaBase {  void meta(const at::Tensor & self, int64_t diagonal);};} // namespace at::musa
```

meta 类的名字和 CPU/CUDA 一样，依靠命名空间实现隔离。需要额外满足情况 3 时，meta 类定义为：

```c
namespace at::musa {

struct TORCH_API structured_tril : public at::musa::MyMetaBase {  template <bool VAR = false>  struct TORCH_API precompute_out {    precompute_out<true> set_var(int64_t value) {      static_assert(VAR == false, "var already set");      precompute_out<true> ret;      ret.var = value;      return ret;    }    int64_t var;  };  using meta_return_ty = precompute_out<true>;  meta_return_ty meta(const at::Tensor & self, int64_t diagonal);};} // namespace at::musa
```

codegen 模块会在 meta 类中生成一个嵌套模板子类 precompute_out，meta 函数返回值由 void 变为该子类的实例化，存储产生的中间变量。impl 函数的 diagonal 参数由 var 代替，而非算子调用时传入的值。因此 impl 类定义为：

```c
namespace at::musa {

struct TORCH_API structured_MusaTril : public at::musa::structured_tril {  void impl(const at::Tensor & self, int64_t var, const at::Tensor & out);};} // namespace at::musa
```

开发者在实现 Customized 形式的 meta/impl 函数时需要注意函数签名和 codegen 生成的接口声明保持一致。以 Tril.cpp 为目标文件，计算逻辑可实现如下：

```c
// 文件位置：torch_musa/csrc/aten/ops/Tril.cpp
namespace at::musa {

TORCH_PRECOMPUTE_META_FUNC(tril)(const Tensor& self, int64_t diagonal) {  // 参数校验
  // 计算临时变量
  int64_t var = ....  // 打包临时变量
  return TORCH_PRECOMPUTE_META_FUNC(tril).set_var(var);}

TORCH_IMPL_FUNC(MusaTril)(const Tensor& self, int64_t var, const Tensor &result) {  // 计算实现
}} // namespace at::musa
```

计算函数的第二个参数是中间变量 var，非算子调用时传入的原始参数 diagonal，剩下的实现过程与 LegacyMeta 方式类似。

总结来看，MUSA structured 算子的开发难度为 Legacy < LegacyMeta < Customized。当 impl 函数使用了 DispatchStub 机制时，我们可以通过 Porting-CUDA 快速实现基础 MUSA kernels；遇到正确性或效率问题时，我们可以结合 MUSA 设备的架构参数，自定义 impl 函数优化计算逻辑；如果要实现全新的预处理策略，最后再考虑自定义 meta 函数，普通情况下一般不会用到。

## 实现UnStructured算子

PyTorch 的 unstructured 算子一般以 functional 规则调用，实现逻辑相互独立，不显式抽象出预处理和计算子逻辑，而是在实现内部自组织。算子定义时，我们需要显式指定 MUSA 后端的派发名，和 structured 算子不同，这个名字就是实现绑定的函数名。以 nonzero 算子为例，参考 CPU/CUDA 声明格式，MUSA 后端可声明如下：

```yaml
# CPU/CUDA 声明，来自 native_functions.yaml
# structured 默认为 false，标记为 unstructured 实现方式
- func: nonzero(Tensor self) -> Tensor
  dispatch:    CPU: nonzero_cpu
    CUDA: nonzero_cuda

# MUSA 声明，来自 musa_functions.yaml
# structured 缺省，默认和官方保持一致
- func: nonzero
  dispatch:    PrivateUse1: Nonzero
```

算子声明可以显式指定 structured 为 false，由于官方实现方式默认是 unstructured，这里也可以忽略不写，codegen 模块自动对齐 CPU/CUDA 的设置。考虑灵活扩展，torch_musa 也支持官方 structured 的实现转换为 MUSA 后端的 unstructured 实现，以 add.Tensor 算子为例（Tensor + Tensor），MUSA 的 unstructured 声明如下：

```yaml
# Unstructured 声明，来自 musa_functions.yaml
- func: add.Tensor
  dispatch:    PrivateUse1: AddTensor
- func: add_.Tensor
  dispatch:    PrivateUse1: AddTensor_
- func: add.out
  structured: false
  dispatch:    PrivateUse1: AddTensorOut
```

如果算子的原始 structured 声明包含下列属性，在 MUSA 声明中需要显式处理：

- structured：out 声明用 false 值覆盖，functional/inplace 声明缺省。
- structured_delegate：functional/inplace 声明用 none 覆盖，out 声明缺省。
- structured_inherits：out 声明用 none 值覆盖，functional/inplace 声明缺省。
- precomputed：out 声明用 none 值覆盖，functional/inplace 声明缺省。
- dispatch：所有规则声明都需要显式指定派发的后端名。

正确设置上述属性后，codegen 模块把 group 内每个调用规则的算子看成是实现独立的，生成注册函数时内部直接调用 MUSA 后端绑定名对应的实现函数，否则解析 yaml 文件会报错。在算子实现部分，我们可以根据实际情况选择最合适的逻辑策略。

### 复用公共函数（Legacy）

当算子的实现逻辑完全不区分后端时，CPU/CUDA 的实现会共用 PyTorch 仓库提供的一个基础函数，内部逻辑可能只涉及非数据计算的视图转换，或者数据计算完全由依次调用其他算子完成。以 view_as_real 算子为例，实现逻辑只是复数拆开成两个浮点数的视图转换，没有数据计算，原始的官方声明如下：

```yaml
- func: view_as_real(Tensor(a) self) -> Tensor(a)
  dispatch:    CPU, CUDA, MPS, Meta: view_as_real
```

可以看出 CPU/CUDA 等后端绑定了一个公共的 view_as_real 函数实现转换；MUSA 后端实现应该保持一致，避免冗余代码，表现在算子声明上：

```yaml
- func: view_as_real
  dispatch:    PrivateUse1: view_as_real
```

在编译时，codegen 模块会在 unstructured 实现方式的前提下对比函数名，判断出 MUSA 的绑定函数和 CPU/CUDA 一样，注册函数内部自动调用对应的公共函数，也称为 Legacy 实现。

### 接入MUDNN（Customized）

如果算子的 MUSA 实现涉及数据计算，正好 MUDNN 库提供了相应的能力时，我们可以在实现逻辑内部直接调用 MUDNN 接口完成计算。使用 MUDNN 库的主要步骤如下：

1. 参数校验，主要检查 device/dtype/defined_tensor 等。
2. 添加 DeviceGuard。
3. 参数转换，比如 conv 算子只支持连续 tensors，需要提前把输入/输出 tensors 转换为连续的。
4. 创建输入/输出 MUTensors，以及 MUDNN 的计算实例，配置计算参数。
5. 调用实例接口完成计算，返回计算结果。

以 add.Tensor 的 functional 算子为例，我们可以在 torch_musa/csrc/aten/ops 目录下创建 Add.cpp 文件，实现 AddTensor 函数：

```cpp
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
```

通过 mudnn\*.h 头文件我们可以查询 MUDNN 库的算子支持情况和接口定义，默认地址为 /usr/local/musa/include 目录。如果 MUDNN 不支持，我们也可以通过 CUDA-Porting 等方式手动实现 MUSA kernels，在函数内部手工调用完成计算。

### CPU计算（Customized）

对于部分算子，如果MUDNN不支持，CUDA-Porting也无法支持，可以临时中转到 CPU 后端实现该算子。主要逻辑是，先把 tensor拷贝到 CPU 上，调用 CPU 算子完成计算，再将结果拷贝回 GPU。可以参考下述代码：

```cpp
Tensor AddTensor(
    const Tensor& self, const Tensor& other, Scalar const& alpha_scalar) {
  const auto cpu_dev = DeviceType::CPU;
  const auto musa_dev = self.device();
  auto cpu_self = at::empty(self.sizes(), self.options().device(cpu_dev));
  auto cpu_other = at::empty(other.sizes(), other.options().device(cpu_dev));
  return at::cpu::add(cpu_self, cpu_other).to(musa_dev);
}
```

在初次适配模型时，可以通过这种方式快速判断缺少哪些算子，然后再逐个适配，通过接入 MUDNN 或自定义 kernels 的方式提高性能。

:::warning 注意：

除了 MUSA 后端（PrivateUse1）外，其他 MUSA 相关后端（比如 QuantizedPrivateUse1）的实现方式一定是 unstructured 的。

:::
:::info 注解：

以上代码仅作参考，不代表实际的实现逻辑。

:::