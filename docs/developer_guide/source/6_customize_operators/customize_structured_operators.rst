实现 Structured 算子
====================================

对于 structured group 内多种调用规则的 C++ 算子，PyTorch 使用多级继承的方式实现统一的预处理和计算逻辑，继承类的命名遵循 torchgen 的规则，添加 “structured” 前缀。以 tril 算子的 CUDA 实现为例，规则如下（由底至上）：

1. 以 functional 算子名创建 meta::structured_tril 类（简称 meta 类，继承官方的 MetaBase 基类），新增 meta 函数实现预处理。
2. 以 out 算子绑定的后端实现名创建 native::structured_tril_cuda 类（简称 impl 类，继承 meta 类），新增 impl 函数实现计算。
3. 创建 structured_tril_cuda_functional 类（简称 functional 类，继承 impl 类），覆写基类方法实现结果 tensor 的创建。
4. 创建 structured_tril_cuda_inplace 类（简称 inplace 类，继承 impl 类），覆写基类方法实现可读写 tensor 的校验。
5. 创建 structured_tril_cuda_out 类（简称 out 类，继承 impl 类），覆写基类方法实现结果 tensor 的校验和尺寸变化。
6. functional/inplace/out 算子实现为分别实例化 3/4/5 中创建的子类，依次调用 meta 和 impl 函数，最后返回结果。

作为开发者，算子的实现部分我们只需完成 meta 类的预处理和 impl 类的计算逻辑开发，实现对应的函数；为了确保 codegen 正确生成算子文件和注册绑定，native_functions.yaml 文件中对三种调用规则的算子定义添加如下字段：

- 不同调用规则的算子类（上述 3/4/5 中的类）继承同一个 impl 类，视为一种 “规约” 关系；impl 类名固定引用 out 算子绑定的后端名，故在 functional/inplace 算子定义中添加 “规约” 字段，目标为 out 算子名。

.. code-block:: yaml

  - func: tril(Tensor self, int diagonal=0) -> Tensor
    # 参考 out 算子的内容，生成 functional 类
    # 标记该算子为 structured 的实现方式
    structured_delegate: tril.out

  - func: tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
    # 参考 out 算子的内容，生成 inplace 类
    # 标记该算子为 structured 的实现方式
    structured_delegate: tril.out

- out 算子声明除了后端名外，需要增加 structured 实现方式的标记，支持扩展字段描述基类信息，以及传递预处理的中间变量映射关系。

.. code-block:: yaml

  - func: tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
    structured: True  # 标记该算子为 structured 的实现方式
    dispatch:
      CUDA: tril_cuda # out 类和 impl 类实现的命名参考
    # 选择要继承的基类，缺省默认为 MetaBase，自定义例如
    # structured_inherits: TensorIteratorBase
    # 预处理传递给计算函数的中间变量映射关系，缺省默认为原参数传递，自定义例如
    # precomputed:
    # - diagonal -> int var

导入上述定义字段后，构建时 torchgen 自动解析字段内容，生成上述 1-6 中所有的类定义和算子实现（调用 meta 和 impl 函数），生成绑定和注册代码，完成算子的适配。

基于上述规范，在实现 structured 的 MUSA 算子时，考虑到 PyTorch 已经集成了 CUDA 的实现，meta/impl 函数可能都可以复用，故有三种实现方法。

复用 meta/impl 函数（Legacy）
------------------

这类算子的 meta 预处理逻辑是通用的，impl 函数内部应用了 PyTorch 的 DispatchStub 机制，即 CPU/CUDA 后端分别把计算 kernel 注册到算子对应的 stub 中，运行时根据目标 device 在 stub 中找到对应的 kernel 完成计算。我们可以直接复用 CUDA 的 impl 实现，只需要实现 MUSA kernel, 并注册到对应的 stub 中即可。

.. note::
    DispatchStub 的原理可参考 pytorch/aten/src/ATen/native/DispatchStub.h 源码文件

由于 MUSA 的编程模型实现了 CUDA 兼容，MUSA kernel 的一种实现方法是通过 CUDA-Porting 工具完成，主要流程如下：

1. 新建目录 build/generated_cuda_compatible，保存 Porting 的 kernels 文件和依赖头文件。
2. 把 PyTorch 仓库中的 CUDA kernels 和安装目录中的头文件复制到新建目录内（维持相对路径）。
3. Porting 工具实施文本替换，如将 cudaMalloc 替换成 musaMalloc，cuda_fp16.h 替换成 musa_fp16.h 等。

上述步骤在编译时依次执行，结束后指定目录下出现目标 mu 文件，包含转换完成的 MUSA kernel 和 stub 注册实现，我们把该文件添加到 musa_kernels 库包含的源文件集合中即可。以 lerp 算子为例，目标文件为 Lerp.mu，包含如下内容：

.. code-block:: yaml

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

可以看到文件中已经包含了计算和注册代码，我们在编译文件中做如下修改：

.. code-block:: cmake

  # 文件位置：/home/torch_musa/torch_musa/csrc/CMakeLists.txt

  file(
    GLOB_RECURSE
    MU_SRCS
    ......
    ${GENERATED_PORTING_DIR}/aten/src/ATen/native/musa/Lerp.mu
    ......
  )

编译完成后 musa_kernels 动态库中会包含 MUSA lerp 的 kernels，在运行时自动注册到 lerp stub 中。
由于 MUSA 和 CUDA 设备的参数和架构存在差异，有时通过 CUDA-Porting 生成的 MUSA kernels 在运行时会报错或者效率不高，此时开发者可以选择手动修改或重写计算逻辑，即放弃在编译文件中加入 Porting 生成的 Lerp.mu 文件，在 torch_musa/csrc/aten/ops/musa 目录下创建 Lerp.mu 文件，手动实现 lerp kernels 和 stub 注册。

对于这种 Legacy 的实现方式，musa_functions.yaml 中算子的定义只需要列出算子名，其他的 structured 关键字 codegen 模块自动与 PyTorch 保持对齐。以 Lerp 算子为例，内容如下：

.. code-block:: yaml

  - func: lerp.Scalar_out
  - func: lerp.Scalar
  - func: lerp_.Scalar

  - func: lerp.Tensor_out
  - func: lerp.Tensor
  - func: lerp_.Tensor

只复用 meta 函数（LegacyMeta）
------------------

当算子的预处理逻辑通用，计算逻辑不使用 DispatchStub 时，开发者需要显式实现 MUSA 后端的 impl 函数。以 tril 算子为例，我们首先在 musa_functions.yaml 文件中添加算子接口定义：

.. code-block:: yaml

  - func: tril
  - func: tril_
  - func: tril.out
    dispatch:
      PrivateUse1: MusaTril

functional/inplace 规则的算子 “规约” 方式保持一致，列出算子名即可；与 Legacy 方式不同，out 算子需要显式指定后端实现名，让 codegen 模块自动生成 impl 类（在 at::musa 命名空间下）。实现计算逻辑时，我们可以在 torch_musa/csrc/aten/ops 目录下新建 Tril.cpp 文件，实现如下函数：

.. code-block:: c++

  // 文件位置：torch_musa/csrc/aten/ops/Tril.cpp
  namespace at::musa {

  TORCH_IMPL_FUNC(MusaTril)(const Tensor& self, int64_t k, const Tensor &result) {
    // 计算过程
  }

  } // namespace at::musa 

PyTorch 针对 structured 类和函数定义了一系列宏，此处 TORCH_IMPL_FUNC(MusaTril) 会自动展开为 void structured_MusaTril::impl，与 codegen 生成的 impl 类函数保持一致。LegacyMeta 的实现方式不用修改编译文件，Tril.cpp 会自动被加入到 musa_kernels 库的源文件集合中。

自定义 meta/impl 函数（Customized）
------------------

如果在实现 MUSA structured 算子时遇到如下情况：

1. 预处理逻辑和 CPU/CUDA 有区别。
2. meta 类需要继承不同的基类。
3. meta 函数需要传给 impl 函数自定义的中间值，与 CPU/CUDA 不同或 CPU/CUDA 不传中间值。

开发者需要同时显式实现 meta 和 impl 函数（都在 at::musa 命名空间下）。以 tril 算子为例，我们首先在 musa_functions.yaml 文件中添加算子接口定义：

.. code-block:: yaml

  - func: tril
  - func: tril_
  - func: tril.out
    structured_inherits: MyMetaBase  # 集成的基类名
    precomputed:
    - diagonal -> int var # 需要传递的中间变量
    dispatch:
      PrivateUse1: MusaTril

out 算子定义中必须显式指定 structured_inherits（情况2）或者 precomputed （情况3）字段。考虑情况 2 ，meta 类的定义如下：

.. code-block:: c++

  namespace at::musa {

  struct TORCH_API structured_tril : public at::musa::MyMetaBase {
    void meta(const at::Tensor & self, int64_t diagonal);
  };

  } // namespace at::musa

meta 类的名字和 CPU/CUDA 一样，依靠命名空间实现隔离。需要额外满足情况 3 时，meta 类定义为：

.. code-block:: c++

  namespace at::musa {

  struct TORCH_API structured_tril : public at::musa::MyMetaBase {

    template <bool VAR = false>
    struct TORCH_API precompute_out {

      precompute_out<true> set_var(int64_t value) {
        static_assert(VAR == false, "var already set");
        precompute_out<true> ret;
        ret.var = value;
        return ret;
      }
                
      int64_t var;
    };

    using meta_return_ty = precompute_out<true>;
    meta_return_ty meta(const at::Tensor & self, int64_t diagonal);
  };

  } // namespace at::musa

codegen 模块会在 meta 类中生成一个嵌套模板子类 precompute_out，meta 函数返回值由 void 变为该子类的实例化，存储产生的中间变量。impl 函数的 diagonal 参数由 var 代替，而非算子调用时传入的值。因此 impl 类定义为：

.. code-block:: c++

  namespace at::musa {

  struct TORCH_API structured_MusaTril : public at::musa::structured_tril {
    void impl(const at::Tensor & self, int64_t var, const at::Tensor & out);
  };

  } // namespace at::musa

开发者在实现 Customized 形式的 meta/impl 函数时需要注意函数签名和 codegen 生成的接口声明保持一致。以 Tril.cpp 为目标文件，计算逻辑可实现如下：

.. code-block:: c++

  // 文件位置：torch_musa/csrc/aten/ops/Tril.cpp
  namespace at::musa {

  TORCH_PRECOMPUTE_META_FUNC(tril)(const Tensor& self, int64_t diagonal) {
    // 参数校验
    // 计算临时变量
    int64_t var = ....
    // 打包临时变量
    return TORCH_PRECOMPUTE_META_FUNC(tril).set_var(var);
  }

  TORCH_IMPL_FUNC(MusaTril)(const Tensor& self, int64_t var, const Tensor &result) {
    // 计算实现
  }

  } // namespace at::musa 

计算函数的第二个参数是中间变量 var，非算子调用时传入的原始参数 diagonal，剩下的实现过程与 LegacyMeta 方式类似。

总结来看，MUSA structured 算子的开发难度为 Legacy < LegacyMeta < Customized。当 impl 函数使用了 DispatchStub 机制时，我们可以通过 Porting-CUDA 快速实现基础 MUSA kernels；遇到正确性或效率问题时，我们可以结合 MUSA 设备的架构参数，自定义 impl 函数优化计算逻辑；如果要实现全新的预处理策略，最后再考虑自定义 meta 函数，普通情况下一般不会用到。
