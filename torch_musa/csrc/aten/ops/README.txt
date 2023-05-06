there ares some ops that implemented on CPU:
clamp.Tensor_out
remainder.Scalar_Tensor
convolution_overrideable   # input.dim() != 4 ||  weight.dim()  (rn50 doesn't hit this case)
cast  # double
bernoulli_.float
uniform_
_index_put_impl_  # index's type == bool (rn50 doesn't hit this case)
scatter.src_out
scatter.src
native_group_norm_backward
min, max # double
mul, div # double, bool
fill # double
