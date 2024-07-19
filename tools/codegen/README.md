## Codegen for MUSA Customized Kernels

Declarations, bindings and registrations of PyTorch aten kernels listed in 
[native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml) are completed by torchgen (see [docs here](https://github.com/pytorch/pytorch/wiki/Codegen-and-Structured-Kernels)). Refer to official [dispatcher_extension](https://pytorch.org/tutorials/advanced/extend_dispatcher.html), torch_musa annotates customized kernels with `PrivateUse1` backend in [musa_functions.yaml](torch_musa/csrc/aten/ops/musa_functions.yaml), and introduces the codegen module based on torchgen to implement bindings and registrations of kernels automatically (declarations are always consistent with PyTorch).

Technically, kernels may be implemented by structured or unstructured strategies based on their logical characteristics (shape changes, type conversions, data dependencies, etc), the implementation in torch_musa show some differences:

- Structured kernels in PyTorch may be implemented by unstructured ways in torch_musa. On the contrary, the structured implementation of PyTorch unstructured kernels is not supported.

- Default namespace is `at::musa`, which is different from `at::native` in PyTorch.

- In some cases, it may be possible to directly invoke the kernel with CPU backend.

The codegen module provides support for the above features and simplifies yaml format for developers.

### Structured kernels

Pytorch splits the implementation of structured kernels into preprocessing and calculation sub-processes, corresponding to the `meta` and `impl` member functions of boilerplate structured classes respectively. Usually, the preprocessing logic is common across backends, while in the calculation stage, each backend should implement specialized acceleration logic based on its architecture parameters and calculation model. Since MUSA programming provides CUDA compatibility, developers can reuse both logics, customize only `impl` or both member functions with different codegen formats.

If MUSA kernels are implemented by porting-CUDA and registered into `{op}_stub`, the actual dispatch happens inside the `impl` function, the outer preprocessing and calculation logics are exactly the same as CPU/CUDA. The corresponding declarations in `native_functions.yaml` only need to list the function names like:

```yaml
- func: bitwise_right_shift.Tensor      # fucntional overload
- func: bitwise_right_shift_.Tensor     # inplace overload
- func: bitwise_right_shift.Tensor_out  # out overload
```

This method is the simplest and called the `Legacy` strategy. Furthermore, if we need to optimize the calculation logic based on MTGPU characteristics, we can just customize the `impl` function and keep the preprocessing consistent, called `LegacyMeta` strategy. Correspondingly the yaml declaration needs to add the explicit `dispatch` key like:

```yaml
- func: bitwise_right_shift.Tensor      # fucntional overload
- func: bitwise_right_shift_.Tensor     # inplace overload
- func: bitwise_right_shift.Tensor_out  # out overload
  dispatch:
    PrivateUse1: musa_bitwise_right_shift_impl
```

Codegen will generate the `impl` member function under `at::musa` namespace for this kernel, and we should complete the development of the accelerated computing codes in this function.

Torchgen introduces `structured_inherits` and `precomputed` keys to control the logic of the preprocessing function, MUSA codegen module inherits them. So if we want to customize all parts of the kernel implementation including preprocessing, we need to add at least one description of the above two keys like:

```yaml
- func: bitwise_right_shift.Tensor      # fucntional overload
- func: bitwise_right_shift_.Tensor     # inplace overload
- func: bitwise_right_shift.Tensor_out  # out overload
  structured_inherits: MyBase           # optional
  precomputed:                          # optional
  - dim -> int dim1, int dim2
  dispatch:
    PrivateUse1: musa_bitwise_right_shift_impl
```

Compared with `LegacyMeta`, codegen will additionally generate the `meta` member function under `at::musa` namespace, the preprocessing logic and return values need to correspond to the yaml declaration format. This fully customized approach is called `Customized` strategy.

The differences between three implementation strategies are as follows:

| Impl Kind | Customize meta(...) | Meta Namespace | Customize impl(...) | Impl Namespace |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Legacy | &#10006; | at::native | &#10006; | at::native |
| LegacyMeta | &#10006; | at::native | &#10004; | at::musa |
| Customized | &#10004; | at::musa |&#10004; | at::musa |


### Unstructured kernels

The implementation of unstructured kernels is completely independent. Considering that torch_musa supports the unstructured implementation of kernels in a PyTorch structured group, we should explicitly specify their `structured` tags as false like:

```yaml
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
```

In native_functions.yaml, `structured` value of `add.Tensor` and `add_.Tensor` kernels are already set false, we can just ignore. For PyTorch unstructured kernels, all we need is to specify the impl function in `dispatch` key with MUSA backend, like:

```yaml
- func: convolution_overrideable
  dispatch:
    PrivateUse1: Convolution
```
By default, unstructured implementation is customizable as well as the binding function names, called `Customized` strategy. Also there are some scenarios for `Legacy` strategy, for example:

```yaml
- func: view_as_real
  dispatch:
    PrivateUse1: view_as_real
```

Some unstructured kernel logisc are backends independent, we can reuse the function name, codegen will prioritize searching for the existence of this function in at::native namespace inside PyTorch official code repository, and additionally bind it to the MUSA backend.
