本节主要介绍如何在torch_musa中适配一个新算子，算子实现后端包括MUDNN算子库和CUDA-Porting kernels。

何时需要适配新算子？
======================

以“tril”算子为例，当我们的测试代码有如下报错log，则说明torch_musa中没有适配“tril”算子。

.. code-block:: python

  import torch
  import torch_musa
  input_data = torch.randn(3, 3, device="musa")
  result = torch.tril(input_data)

.. figure:: ../doc_image/not_implemented.*

如何适配新算子
==============

先注册新算子
-------------

算子实现的注册可以参考PyTorch官方文档https://pytorch.org/tutorials/advanced/dispatcher.html ，也可以参考PyTorch框架中CUDA后端的注册代码。在编译完PyTorch代码后会生成下图中的文件，PyTorch在该文件中实现了CUDA后端实现的注册。

.. figure:: ../doc_image/RegisterCUDA.*

同理，我们也需要给“tril”算子为MUSA后端实现注册。部分代码如下所示：

.. code-block:: c++

  #include <torch/library.h>
  
  namespace at {
  namespace musa {
  
  TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("tril", Tril);  // 'Tril' is a function implemented somewhere
  }
  
  } // namespace musa
  } // namespace at


PyTorch社区推荐使用PrivateUse1作为第三方扩展后端的key，所以我们这里复用了PrivateUse1。

利用MUDNN实现算子
------------------

如果MUDNN算子库支持了该算子，那么需要以MUDNN算子库作为后端来适配该新算子。使用MUDNN适配新算子的主要步骤如下：

#. tensor的数据类型和device类型检查;
#. 添加DeviceGuard;
#. 目前大部分MUDNN算子只支持连续的tensor，因此在 ``createMUTensor`` 前需要将其转换为连续tensor（如果该tensor不连续，此操作将会产生copy耗时）；
#. 创建muTensor；
#. 调用MUDNN的op实现接口；

以 ``addcdiv.out`` 算子为例，部分代码如下：

.. code-block:: c++

  #include <mudnn.h>
  
  Tensor& AddcDivOut(const Tensor& base, const Tensor& tensor1,
                     const Tensor& tensor2, const Scalar& value, Tensor& out) {
    //1). check Dtype & device
    TORCH_CHECK(self.device().type() == kMUSA,
                "Device of input tensor of addcdiv must be MUSA, but now it is ",
                self.device());
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "Dtype of input tensor of addcdiv only support Float32, but now it is ",
        self.scalar_type());
       ....
    // 2).convert it to contiguous tensor
    Tensor tensor_cong = tensor1.contiguous();
       ...
    // 3). create muTensor，which binds the two variables by address.
    muTensor musa_tensor1 = CreateMUTensor(tensor_cong);
    muTensor mu_out = CreateMUTensor(tensor_cong);     ....
    // 4). call musa op to implement the calculation.
    ::musa::dnn::Handle h;
    ::musa::dnn::Ternary mop;
  
    if (!alpha_scalar.equal(1)) {
      if (self.is_floating_point()) {
        CHECK_MUDNN_STATUS(mop.SetAlpha(alpha_scalar.toDouble()), "SetAlpha");
      } else {
        CHECK_MUDNN_STATUS(mop.SetAlpha(alpha_scalar.toLong()), "SetAlpha");
      }
    }
    CHECK_MUDNN_STATUS(mop.SetMode(TERNARY_MODE::ADDCDIV_ALPHA), "SetMode");
    CHECK_MUDNN_STATUS(mop.Run(h, om_mt, musa_base, musa_tensor1, musa_tensor2), "Run");
  
  }
  
  TORCH_LIBRARY_IMPL(aten, PrivateUse1, m){
     ...
     m.impl("addcdiv.out", &AddcDivOut);
  }

通过mudnn*.h头文件可以查看到MUDNN算子库函数接口。默认MUDNN算子库的头文件会在 ``/usr/local/musa/include`` 目录下。


利用CUDA-Porting实现算子
------------------------

如果该算子MUDNN算子库不支持，那么我们需要通过CUDA-Porting kernels作为后端来适配新算子。

首先介绍一下CUDA-Porting的流程：

#. 在torch_musa/build下新建目录（默认目录名是torch_musa/build/generated_cuda_compatible）用来保存CUDA-Porting过程需要用到的文件。
#. 从PyTorch仓库中将kernels相关的cu/cuh文件以及include头文件复制到上一步新建目录中去。这些文件需要经过CUDA-Porting脚本的处理（torch_musa/torch_musa/tools/cuda_porting/cuda_porting.py）。
#. 运行porting工具。主要是一些字符串替换处理，如将cudaMalloc替换成musaMalloc，cuda_fp16.h替换成musa_fp16.h等。
#. 经过上述操作后，build/generated_cuda_compatible/aten/src/ATen/native/musa/会有很多****.mu文件，这些mu文件就是我们适配时会用到的kernels文件。
#. 适配CUDA-Porting工具处理过的kernels。

上述步骤1，2，3，4会在编译过程中自动完成，适配新算子关心的步骤5即可。有一点需要注意的是，在开发过程中引用的PyTorch头文件来自于 ``torch_musa/build/generated_cuda_compatible/include`` 目录，而不是系统下PyTorch安装目录下的头文件。


下面以两种典型算子为例，介绍如何利用CUDA-Porting kernels适配新算子。在开始适配之前，可以在 ``pytorch/build/aten/src/ATen/RegisterCUDA.cpp`` 文件中查看该算子在CUDA中的实现方式。

以abs算子为例
^^^^^^^^^^^^^^

CUDA中abs算子的部分适配代码如下：

.. code-block:: c++

  at::Tensor & wrapper_CUDA_out_abs_out(const at::Tensor & self, at::Tensor & out) {
    // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::abs_out(self, out);
  }
  
  ******
  m.impl("abs.out", TORCH_FN(wrapper_CUDA_out_abs_out));

如果该算子直接调用了at::native下面的函数接口，那么我们也这么做就可以了：

.. code-block:: c++

  #include "torch_musa/csrc/core/MUSAGuard.h"
  at::Tensor& MusaAbsout(const at::Tensor& self, at::Tensor& out) {
  c10::musa::MUSAGuard device_gaurd(self.device());
  return at::native::abs_out(self, out);
  }

  TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("abs.out", &MusaAbsout);
  }

这里的关键是PyTorch仓库提供了DispatchStub机制。我们在CUDA-Porting时，将 ``REGISTER_CUDA_DISPATCH`` 替换成 ``REGISTER_MUSA_DISPATCH`` ，从而能实现根据device类型调用到porting后的kernels。对这背后机制感兴趣的话，可以查看一下如下几个文件：

- abs_out函数实现：https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/UnaryOps.cpp#L546
- abs_stub注册：https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/cuda/AbsKernel.cu#L49
- DispatchStub定义：https://github.com/pytorch/pytorch/blob/v2.0.0/aten/src/ATen/native/DispatchStub.h

以tril算子为例
^^^^^^^^^^^^^^^

CUDA中tril算子的部分适配代码如下：

.. code-block:: c++

  struct structured_tril_cuda_functional final : public at::native::structured_tril_cuda {
      void set_output_strided(
          int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
          TensorOptions options, DimnameList names
      ) override {
          auto current_device = guard_.current_device();
          if (C10_UNLIKELY(current_device.has_value())) {
            TORCH_INTERNAL_ASSERT(*current_device == options.device(),
              "structured kernels don't support multi-device outputs");
          } else {
            guard_.reset_device(options.device());
          }
          outputs_[output_idx] = create_out(sizes, strides, options);
          if (!names.empty()) {
            namedinference::propagate_names(*outputs_[output_idx], names);
          }
          // super must happen after, so that downstream can use maybe_get_output
          // to retrieve the output
      }
      void set_output_raw_strided(
          int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
          TensorOptions options, DimnameList names
      ) override {
          auto current_device = guard_.current_device();
          if (C10_UNLIKELY(current_device.has_value())) {
            TORCH_INTERNAL_ASSERT(*current_device == options.device(),
              "structured kernels don't support multi-device outputs");
          } else {
            guard_.reset_device(options.device());
          }
          outputs_[output_idx] = create_out(sizes, strides, options);
          if (!names.empty()) {
            namedinference::propagate_names(*outputs_[output_idx], names);
          }
          // super must happen after, so that downstream can use maybe_get_output
          // to retrieve the output
      }
      const Tensor& maybe_get_output(int64_t output_idx) override {
        return *outputs_[output_idx];
      }
      std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
      c10::cuda::OptionalCUDAGuard guard_;
  };
  at::Tensor wrapper_CUDA_tril(const at::Tensor & self, int64_t diagonal) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
    c10::impl::check_and_update_common_device(common_device, self, "wrapper_CUDA_tril", "self");
    structured_tril_cuda_functional op;
  op.meta(self, diagonal);
  op.impl(self, diagonal, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
  }

  ******
  m.impl("tril", TORCH_FN(wrapper_CUDA_tril));

该算子在实现时继承了基类 ``at::native::structured_tril_cuda`` , 那么我们也需要这么实现：

.. code-block:: c++

  #include <ATen/ops/tril_native.h>

  #include "torch_musa/csrc/aten/utils/Utils.h"
  #include "torch_musa/csrc/core/MUSAGuard.h"
  
  namespace at {
  namespace musa {
  
  namespace {
  struct structured_tril_musa_functional final
      : public at::native::structured_tril_cuda {
    void set_output_strided(
        int64_t output_idx,
        IntArrayRef sizes,
        IntArrayRef strides,
        TensorOptions options,
        DimnameList names) override {
      auto current_device = guard_.current_device();
      if (C10_UNLIKELY(current_device.has_value())) {
        TORCH_INTERNAL_ASSERT(
            *current_device == options.device(),
            "structured kernels don't support multi-device outputs");
      } else {
        guard_.reset_device(options.device());
      }
      outputs_[output_idx] = create_out(sizes, strides, options);
    }
    void set_output_raw_strided(
        int64_t output_idx,
        IntArrayRef sizes,
        IntArrayRef strides,
        TensorOptions options,
        DimnameList names) override {
      auto current_device = guard_.current_device();
      if (C10_UNLIKELY(current_device.has_value())) {
        TORCH_INTERNAL_ASSERT(
            *current_device == options.device(),
            "structured kernels don't support multi-device outputs");
      } else {
        guard_.reset_device(options.device());
      }
      outputs_[output_idx] = create_out(sizes, strides, options);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
      return *outputs_[output_idx];
    }
    std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
    c10::musa::OptionalMUSAGuard guard_;
  };
  } // namespace
  
  Tensor Tril(const Tensor& self, int64_t diagonal) {
    structured_tril_musa_functional op;
    op.meta(self, diagonal);
    op.impl(self, diagonal, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
  }
  } // namespace musa
  } // namespace at

至此，我们已经完成了通过CUDA-Porting kernels来适配新算子。

利用CPU实现算子
------------------------

对于部分算子，如果MUDNN不支持，CUDA-Porting也无法支持，可以临时中利用CPU后端实现该算子。主要逻辑是，先把tensor拷贝到CPU侧，在CPU完成计算，再将结果拷贝到GPU侧。可以参考下述代码：

.. code-block:: c++

  Tensor& AddcDivOut(const Tensor& base, const Tensor& tensor1,
                       const Tensor& tensor2, const Scalar& value, Tensor& out) {
    auto cpu_base =
        at::empty(base.sizes(), base.options().device(DeviceType::CPU));
    auto cpu_factor1 =
        at::empty(tensor1.sizes(), tensor1.options().device(DeviceType::CPU));
    auto cpu_factor2 =
        at::empty(tensor2.sizes(), tensor2.options().device(DeviceType::CPU));
    auto cpu_out =
        at::empty(out.sizes(), out.options().device(DeviceType::CPU));
    cpu_base.copy_(base);
    cpu_factor1.copy_(tensor1);
    cpu_factor2.copy_(tensor2);
    auto result = addcdiv_out(cpu_out, cpu_base, cpu_factor1, cpu_factor2);
    out.copy_(cpu_out);
    return out;
  }

添加算子单元测试
-----------------

如果已经完成了新算子的适配，那么还需要添加算子单元测试，保证算子适配结果的正确性。算子测试文件在 ``torch_musa/tests/unittest/operator`` 目录下，参考已有算子测试添加即可，在此不展开描述。

算子测试命令如下：

.. code-block:: bash

  pytest -s torch_musa/tree/main/tests/unittest/operator/xxxx.py


使用调试注册接口注册算子
------------------------

如果希望对算子的调用记录和输入输出值加以记录或统计，可以使用调试注册接口进行注册。

调试注册接口是一组宏，提供对算子的额外调试封装。目前提供三个宏作为注册接口使用：

1. ADVANCED_REGISTER

ADVANCED_REGISTER(lib, key, yaml, func) 是对原生注册机制TORCH_LIBRARY_IMPL的改进版本。

默认情况下，它将原算子封装为一个包含了额外调试组件的新函数，并附加上额外的隐藏前缀注册。

使用时，将原生注册时指定的lib（如aten）、key（如PrivateUse1）、算子的注册名称（如"Abs"）和算子实现函数作为参数传入即可。

下面是一个示例：

原注册方式：

.. code-block:: c++

  TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("Abs", &CustomAbs);
  }

新的注册方式：

.. code-block:: c++

  ADVANCED_REGISTER(aten, PrivateUse1, "Abs", CustomAbs)

在上述示例中，ADVANCED_REGISTER会首先将CustomAbs封装为一个别名函数，并在实际注册时，使用这个封装后的函数进行注册。

默认情况下，这个别名函数的名称为wrapper_CustomAbs。

如果函数名称中存在额外的名字空间，或该函数为类成员函数等情况时，函数名称可能包含特殊符号（如::等）。此时需要使用下述的REGISTER_IMPL指定封装名称。

2. REGISTER_IMPL

REGISTER_IMPL(lib, key, yaml, func, name)的用法与ADVANCED_REGISTER相似。

但与之不同的是，REGISTER_IMPL额外接受一个name参数，以指定封装后的别名函数的命名：该别名函数将被命名为wrapper\_##name。

如果name和func参数相同时，REGISTER_IMPL即为ADVANCED_REGISTER。

下面给出一个示例：

.. code-block:: c++

  REGISTER_IMPL(aten, PrivateUse1, "view", at::native::view, at_native_view)

示例中，view算子的实现为at::native::view，包含了名字空间，因此使用REGISTER_IMPL将其名称指定为at\_native\_view。

其实际注册的封装函数名称为wrapper\_at\_native\_view。

3. REDEFINE_REGISTER

部分情况下，同一个算子可能会被注册为多个不同的别名，例如，not_equal和ne都使用相同的实现。

在已经使用ADVANCED_REGISTER或REGISTER_IMPL将其封装为别名函数之后，不能再进行重复封装，而应直接复用封装后的函数。

因此，使用REDEFINE_REGISTER(lib, key, yaml, name)进行重复注册。

如果之前的封装注册使用ADVANCED_REGISTER，则name即为ADVANCED_REGISTER中的func参数。

如果之前的封装注册使用REGISTER_IMPL，则name为REGSITER_IMPL中的name参数。

示例如下：

.. code-block:: c++
  
  ADVANCED_REGISTER(aten, PrivateUse1, "ne.Tensor", NotEqualTensor)
  ADVANCED_REGISTER(aten, PrivateUse1, "ne_.Tensor", NotEqual_Tensor)
  ADVANCED_REGISTER(aten, PrivateUse1, "ne.Tensor_out", NotEqual_out)
  // not_equal, alias for torch.ne
  REDEFINE_REGISTER(aten, PrivateUse1, "not_equal.Tensor", NotEqualTensor)
  REDEFINE_REGISTER(aten, PrivateUse1, "not_equal_.Tensor", NotEqual_Tensor)
  REDEFINE_REGISTER(aten, PrivateUse1, "not_equal.Tensor_out", NotEqual_out)

示例中，首先使用ADVANCED_REGISTER将NotEqual系列的三个实现分别注册为ne类接口，之后的not_equal类接口使用了相同的实现，因此，改用REDEFINE_REGISTER注册。

即将支持的特性
-----------------

调试注册接口将在未来支持torch原生的函数封装宏，如TORCH_FN等，可以适应更多的算子注册方式。

引入codegen模块，实现算子的注册代码和实现代码的生成，能进一步简化算子适配的工作量。

请关注这部分工作。
