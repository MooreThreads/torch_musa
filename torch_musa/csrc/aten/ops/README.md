there ares some ops that implemented on CPU:
 - clamp.Tensor_out
 - remainder.Scalar_Tensor
 - convolution_overrideable   # input.dim() != 4 ||  weight.dim()  (rn50 doesn't hit this case)
 - cast  # double
 - bernoulli_.float
 - uniform_
 - \_index_put_impl_  # index's type == bool (rn50 doesn't hit this case)
 - scatter.src_out
 - scatter.src


## Use device guard rightly

You should use guard(MUSAGuard/OptionalMUSAGuard) to set specified device id in operators enter point. e.g.

```
Tensor NativeDropoutBackward(const Tensor& grad_output, const Tensor& mask, double scale) {
  ...
  torch_musa::MUSAGuard device_guard(input.device()); 
  ...
  muHandle& h = GetMudnnHandle();
}

  m.impl("native_dropout", &NativeDropout);
```
Note: set device guard in `NativeDropout` then substack will in `input.device()` context.


It's not necessary to use guard if your operator don't depend on device context. e.g.
```
at::Tensor& RandomFrom(at::Tensor& self, ...) {
  Device device = self.device();
  ...
  a1 = a1.to(device);
  return a1;
}
  m.impl("random_.from", &RandomFrom);
```
This operator don't depend on muDNN so no call `GetMudnnHandle`. Just use self device info.

Note: advise to use `at::empty_like` with options that include device info to create a tensor, e.g.
  ```
  at::empty_like(input, input.options())
  ```
Don't advise to use `empty_mtgpu` like this,
  ```
  Tensor result = empty_mtgpu(
    {mat1.size(0), mat2.size(1)},
    self.scalar_type(),
    c10::nullopt,
    kMUSA,
    c10::nullopt,
  ```
This tensor will loss device index info if no set device guard.
