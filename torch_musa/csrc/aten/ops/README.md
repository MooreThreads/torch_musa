
## Use device guard rightly

You should use guard(MUSAGuard/OptionalMUSAGuard) to set specified device id in operators enter point. e.g.

```
Tensor NativeDropoutBackward(const Tensor& grad_output, const Tensor& mask, double scale) {
  ...
  c10::musa::MUSAGuard device_guard(input.device()); 
  ...
  muHandle& h = GetMudnnHandle();
  ...
}

  m.impl("native_dropout", &NativeDropout);  // or ADVANCED_REGISTER(...), see torch_musa/csrc/utils/register_wrapper.h
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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("random_.from", &RandomFrom);
  ...
}
```
This operator doesn't depend on muDNN so no call `GetMudnnHandle`. Just use self device info.

Note: advise to use `at::empty_like` with options that include device info to create a tensor, e.g.
  ```
  at::empty_like(input, input.options())
  ```
It's better not to use `empty_musa` like this,
  ```
  Tensor result = empty_musa(
    {mat1.size(0), mat2.size(1)},
    self.scalar_type(),
    c10::nullopt,
    kMUSA,
    c10::nullopt,
  ```
This tensor will loss device index info if no set device guard.
