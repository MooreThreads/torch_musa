# Fused Optimizers for MUSA backend

We have implemented some fused optimizers based on the MUSA backend software and hardware characteristics. The currently supported fused optimizers are listed bellow. These implementations fully leverage the parallel computing capabilities and memory access features of MUSA hardware, resulting in significant performance improvements compared to standard PyTorch implementations (i.e., porting CUDA's implementations directly).  
Besides, `FusedAdam` and `FusedAdamW` also support DTensor and other Tensor variants that based on the `__torch_dispatch__` mechanism, ensuring compatibility with PyTorch's Tensor parallelism and FP8 training.
  
- FusedLAMB
- FusedAdam
- FusedAdamW
  

## Usage
In PyTorch, users can enable the fused optimizer by specifying `fused=True` in the constructor of the optimizer, for example: `Adam(fused=True)`. However, when using `torch_musa`, this native PyTorch approach may result in suboptimal performance. Instead, we recommend using the fused optimizers implemented in `torch_musa`, which are optimized for MUSA backend. They can be used in the following way:  
  
```python
from torch_musa.optim import FusedAdamW

optimizer = FusedAdamW(model.parameters(), lr=0.1, momentum=0.9)
optimizer.zero_grad()
loss_fn(model(input), target).backward()
optimizer.step()
```