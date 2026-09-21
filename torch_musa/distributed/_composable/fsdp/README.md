# FSDP2 Computation/Communication Overlap on MUSA

This directory contains the `torch_musa` patches for PyTorch composable FSDP2
(`torch.distributed.fsdp.fully_shard`). The patches keep the FSDP2 user API
unchanged, but can replace the all-gather and reduce-scatter communication
paths and adjust the MUSA streams used for FSDP2 computation/communication
overlap.

The optimization is mainly for intra-node FSDP2 training. It targets parameter
all-gather in forward/backward prefetch paths and gradient reduce-scatter in
backward, so that communication can overlap better with model computation.

## Recommended Usage

Most users only need to set `TORCH_MUSA_FSDP2_COMM_TYPE`.

```bash
export TORCH_MUSA_FSDP2_COMM_TYPE=1
unset TORCH_MUSA_FSDP2_OVERLAP_LEVEL
python train.py
```

`TORCH_MUSA_FSDP2_COMM_TYPE=1` enables the low-contention symmetric-memory
all-gather/reduce-scatter implementation. This is the recommended first choice
for FSDP2 overlap optimization, and it generally has better memory consumption
than `TORCH_MUSA_FSDP2_COMM_TYPE=2`.

`TORCH_MUSA_FSDP2_OVERLAP_LEVEL` is an advanced tuning/debugging knob. Most
users should leave it unset; when `TORCH_MUSA_FSDP2_COMM_TYPE` is set to `1` or
`2` on supported hardware, `torch_musa` automatically selects the overlap mode
intended for the custom communication path.

## Import Order

Set the environment variables before importing `torch_musa`. The safest pattern
is to set them in the shell before starting Python. The FSDP2 overlap policy is
resolved while the `torch_musa` FSDP2 patch is installed, so changing these
environment variables later in the same process may not update the selected
overlap policy.

Import `torch_musa` before importing `fully_shard`, so that the FSDP2 patch is
installed before `fully_shard` is bound in user code:

```python
import torch
import torch_musa  # applies torch_musa distributed/FSDP2 patches
from torch.distributed.fsdp import fully_shard

# Build the model, initialize process groups/device meshes, then call fully_shard
# exactly as in regular PyTorch FSDP2 code.
```

If `fully_shard` was imported before `torch_musa`, restart the process or
re-import it after importing `torch_musa`.

## Environment Variables

### `TORCH_MUSA_FSDP2_COMM_TYPE`

This variable chooses the FSDP2 all-gather/reduce-scatter communication
implementation.

| Value | Behavior |
| --- | --- |
| unset or `0` | Use the default PyTorch FSDP2 communication objects. No custom FSDP2 communication is enabled. |
| `1` | Use low-contention symmetric-memory all-gather/reduce-scatter ops. Recommended for most users who want FSDP2 overlap optimization. Usually uses less memory than `2`. |
| `2` | Use MCCL all-gather/reduce-scatter with communication buffers allocated from a symmetric memory pool. Useful for comparison or when this path performs better for a workload. |

Custom communication is enabled only on MUSA arch `mp_31` or newer. On older
architectures, the custom communication path is disabled.

If `TORCH_MUSA_FSDP2_OVERLAP_LEVEL` is unset, setting
`TORCH_MUSA_FSDP2_COMM_TYPE=1` or `2` on supported hardware also selects the
default overlap policy for the custom communication path:
`OVERLAP_FSDP_COMM_COPY_IN_WITH_COPY_OUT` (`2`).

### `TORCH_MUSA_FSDP2_OVERLAP_LEVEL`

This variable explicitly selects the FSDP2 stream-overlap policy. It has higher
priority than the automatic policy selected from `TORCH_MUSA_FSDP2_COMM_TYPE`.
Most users do not need to set it.

| Value | Policy |
| --- | --- |
| unset | Use the automatic policy described below. |
| `0` | `NO_OVERLAP`: use the current stream for FSDP2 communication and computation. |
| `1` | `OVERLAP_FSDP_COMM_ONLY`: move FSDP communication to a high-priority stream. This mode only overlaps effectively when the user sets an explicit FSDP prefetch order. |
| `2` | `OVERLAP_FSDP_COMM_COPY_IN_WITH_COPY_OUT`: overlap the next all-gather copy-in with the previous copy-out/compute path. This is the automatic choice when custom communication is enabled. |
| `3` | `OVERLAP_FSDP_COMM_COPY_IN_WITH_COMM`: use separate high-priority streams for all-gather copy-in and all-gather communication. |
| `4` | `OVERLAP_HSDP_COMM`: use a PyTorch-FSDP2-like overlap policy, including independent streams for all-gather, reduce-scatter, and HSDP all-reduce. |

When `TORCH_MUSA_FSDP2_OVERLAP_LEVEL` is unset, `torch_musa` resolves the
overlap policy as follows:

1. If `TORCH_MUSA_FSDP2_DISABLE_OVERLAP=0`, use `OVERLAP_HSDP_COMM` (`4`).
   This compatibility variable is deprecated.
2. Else, if `TORCH_MUSA_FSDP2_COMM_TYPE` is `1` or `2` and the MUSA arch is
   `mp_31` or newer, use `OVERLAP_FSDP_COMM_COPY_IN_WITH_COPY_OUT` (`2`).
3. Else, use `NO_OVERLAP` (`0`).

### `TORCH_MUSA_FSDP2_DISABLE_OVERLAP`

This is a deprecated compatibility variable. Do not use it in new setups.
When `TORCH_MUSA_FSDP2_OVERLAP_LEVEL` is unset, setting
`TORCH_MUSA_FSDP2_DISABLE_OVERLAP=0` enables `OVERLAP_HSDP_COMM` (`4`).

## Common Configurations

Baseline behavior:

```bash
unset TORCH_MUSA_FSDP2_COMM_TYPE
unset TORCH_MUSA_FSDP2_OVERLAP_LEVEL
python train.py
```

Recommended custom overlap optimization:

```bash
export TORCH_MUSA_FSDP2_COMM_TYPE=1
unset TORCH_MUSA_FSDP2_OVERLAP_LEVEL
python train.py
```

Try the MCCL symmetric-memory-pool communication path:

```bash
export TORCH_MUSA_FSDP2_COMM_TYPE=2
unset TORCH_MUSA_FSDP2_OVERLAP_LEVEL
python train.py
```

Force a specific overlap policy for debugging:

```bash
export TORCH_MUSA_FSDP2_COMM_TYPE=1
export TORCH_MUSA_FSDP2_OVERLAP_LEVEL=2
python train.py
```

Only set `TORCH_MUSA_FSDP2_OVERLAP_LEVEL` when you need to compare or debug a
specific overlap mode. For regular use, prefer selecting the communication path
with `TORCH_MUSA_FSDP2_COMM_TYPE` and leave the overlap level unset.
