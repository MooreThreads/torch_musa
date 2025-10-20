# torch_musa Unified Memory Management

## Overview

M1000 architecture employs a UMA (Unified Memory Addressing) design, enabling both GPU and CPU to access a single, shared physical memory space.

To optimize memory consumption during model execution on M1000, this implementation enables:
- Elimination of duplicate memory allocation on GPU
- Reduction of memory copy between host and device
- Direct GPU access to memory originally allocated by CPU allocator

We propose Unified Memory Management support for the MUSA backend, which avoids GPU memory allocation in `torch.load(map_location="musa")`.

## Usage

Two invocation methods are supported.

### Method 1: Global Configuration via Environment Variable
```bash
export PYTORCH_MUSA_ALLOC_CONF="cpu:unified"
```

### Method 2: Context Manager
```python
import torch_musa
with torch_musa.use_unified_cpu_allocator(): 
   # your code 
```

## Effect
By using Unified Memory Management for the MUSA backend, torch.load(map_location="musa") eliminates the need for GPU memory allocation and memory copy between host and device.