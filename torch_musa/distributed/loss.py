"""Implement FusedCrossEntropyLossTensorParallel, see the docstring for more details"""

# pylint: disable=all
from functools import partial

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.experimental._func_map import local_map

from torch_musa.utils import ext_loader


ext_module = ext_loader.load_ext("_ext", ["online_softmax", "cross_entropy_loss"])

__all__ = ["parallel_fused_cross_entropy"]


# Thanks the Triton implementation provided by https://github.com/NVIDIA/TransformerEngine/pull/1456
# I found triton kernel launch faild on S4000, so use MUSA C++ implementation instead
class FusedCrossEntropyLossTensorParallel(torch.autograd.Function):
    """
    This class implements cross entropy loss which allows DTensor input. The input tensor should
    be in FP32 dtype (loss and gradients are calculated in FP32).

    This implementation also works on a single device, i.e., no tensor parallel, but may have
    less efficiency than F.cross_entropy on arch >= mp_31.

    TODO(mingyuan.wang): support label_smoothing > 0.0

    Examples::

        >>> # Example on tensor parallel
        >>> device_mesh = init_device_mesh('musa', (2,))
        >>> input = torch.randn(4, 1024, device='musa', requires_grad=True)
        >>> target = torch.randint(1024, (4,), device='musa)
        >>> dist_input = distribute_tensor(input, device_mesh, placements=[Shard(1)]
        >>> loss = parallel_fused_cross_entropy(dist_input, target, pg=device_mesh.get_group())
        >>> loss.backward()
        >>>
        >>> # Example on single device
        >>> input = torch.randn(4, 1024, device='musa', requires_grad=True)
        >>> target = torch.randint(1024, (4,), device='musa)
        >>> loss = parallel_fused_cross_entropy(input, target)
        >>> loss.backward()
    """

    @partial(
        local_map,
        out_placements=[Replicate()],
        in_placements=(None, [Shard(1)], None, None, None),
    )
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
        pg: dist.ProcessGroup = None,
    ) -> torch.Tensor:
        """Forward pass of parallel_fused_cross_entropy kernel

        The forward computation is divided into three steps:
            1. use online_softmax to compute max, expsum and the logits value(correspond to the target index)
            2. all-gather step1's computation results
            3. run the final calculation of loss and compute the gradients meanwhile

        Arguments:
            input (torch.Tensor): Predicted unnormalized logits, the shape of input should be (N, C).
            target (torch.Tensor): Ground truth class indices, the shape of target should be (N,).
            reduction (str, optional): Specifies the reduction to apply to the output, only 'mean' and 'sum' are supported currently.
            pg (torch.dist.ProcessGroup): the communication process group of tensor parallel, None if no TP used.
        """
        assert reduction in [
            "mean",
            "sum",
        ], f"FusedCrossEntropyLossTensorParallel only support mean and sum reduction mode, but got {reduction} "

        if pg:
            rank = dist.get_rank(pg)
            world_size = dist.get_world_size(pg)
        else:
            rank = 0
            world_size = 1

        # partial_max_expsum_y.shape: (num_tokens, 3)
        partial_max_expsum_y = ext_module.online_softmax(input, target, rank)

        if world_size > 1:
            gathered_max_expsum_y = funcol.all_gather_tensor(
                partial_max_expsum_y, gather_dim=1, group=pg
            )
        else:
            gathered_max_expsum_y = partial_max_expsum_y
        loss = ext_module.cross_entropy_loss(
            input,
            target,
            gathered_max_expsum_y,
            rank,
            world_size,
            reduction,
        )

        ctx.save_for_backward(input.detach())

        return loss

    @partial(
        local_map,
        out_placements=([Shard(1)], None, None, None),
        in_placements=(None, [Replicate()]),
    )
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass of parallel_fused_cross_entropy kernel, non-op if last layer"""

        (input,) = ctx.saved_tensors

        if torch.equal(grad_output, torch.ones_like(grad_output)):
            pass
        else:
            # PyTorch will complaint inplace modification error during backward if
            # use `input.mul_(grad_output)`
            input = torch.mul(input, grad_output)

        return input, None, None, None


def parallel_fused_cross_entropy(input, target, reduction: str = "mean", pg=None):
    """A wrapper of FusedCrossEntropyLossTensorParallel class, which workarounds the
    error: 'TypeError: apply() takes no keyword arguments' when using keyword arguments,
    see https://github.com/pytorch/pytorch/issues/16940
    """

    return FusedCrossEntropyLossTensorParallel.apply(input, target, reduction, pg)
