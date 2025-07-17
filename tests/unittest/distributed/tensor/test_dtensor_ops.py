"""Test operators with DTensor inputs"""

# pylint: disable=invalid-name
import warnings
import itertools

import torch
from torch.overrides import resolve_name
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map

from torch.testing._internal.distributed._tensor.common_dtensor import DTensorConverter
from torch.distributed._tensor import (
    DTensor,
    distribute_tensor,
    Replicate,
    Shard,
)

from torch_musa.testing.common_fsdp import FSDPTest


NUM_DEVICES = 2


# rename FSDPTest to MultiProcessesTest ?
class DTensorOpTestBase(FSDPTest):
    """Base class of DTensorOPTest"""

    @property
    def world_size(self) -> int:
        return NUM_DEVICES


class TestDTensorOps(DTensorOpTestBase):
    """consists of test suits of DTensorOps"""

    def test_fused_rmsnorm(self):
        """test fused rmsnorm, both forward and backward"""
        self.mesh = self._device_mesh_plan()
        device = torch.musa.current_device()

        def fused_rmsnorm_fwd_bwd(input_, normalized_shape, eps, weight):
            output, inv_var = torch.ops.aten._fused_rmsnorm_forward(
                input_, normalized_shape, eps, weight
            )
            grad_output = torch.ones_like(input_)
            if isinstance(input_, DTensor):
                grad_output = distribute_tensor(
                    grad_output,
                    device_mesh=input_.device_mesh,
                    placements=input_.placements,
                )
            grad_input, grad_weight = torch.ops.aten._fused_rmsnorm_backward(
                grad_output, inv_var, input_, normalized_shape, eps, weight
            )

            return output, grad_input, grad_weight

        # input, normalized_shape, eps, weight
        args = [
            torch.randn(4, 4096, 4096, device=device),
            (4096,),
            1e-6,
            torch.randn(4096, device=device),
        ]
        kwargs = {}

        to_dtensor = DTensorConverter(self.mesh, args, kwargs)
        # other combinations also works, but we test most frequently used cases here
        choices_for_args = [
            [Shard(dim=0), Shard(dim=1)],
            [
                Replicate(),
            ],
        ]
        to_dtensor.sharding_combs = iter(itertools.product(*choices_for_args))

        self.run_dtensor_crossref(
            fused_rmsnorm_fwd_bwd, args, kwargs, to_dtensor=to_dtensor
        )

    def assert_ref_dtensor_equal(self, dtensor_rs, rs):
        """check single device semantics"""
        flat_dtensor_rs = pytree.tree_leaves(dtensor_rs)
        flat_rs = pytree.tree_leaves(rs)
        assert len(flat_dtensor_rs) == len(flat_rs)
        for dtensor_r, r in zip(flat_dtensor_rs, flat_rs):
            if not isinstance(r, torch.Tensor):
                continue
            assert isinstance(dtensor_r, torch.Tensor)

            if self.rank == 0:
                assert (
                    dtensor_r.shape == r.shape
                ), f"Shape mismatch! original shape:{r.shape}, dtensor shape: {dtensor_r.shape}"
                assert (
                    dtensor_r.requires_grad == r.requires_grad
                ), "op result requires_grad mismatch!"
                # import numpy as np
                # np.testing.assert_allclose(dtensor_r.cpu().numpy(), r.cpu().numpy())
                assert torch.allclose(dtensor_r, r, atol=1e-4, rtol=1e-4)

    def run_dtensor_crossref(self, func, args, kwargs, to_dtensor=None):
        """run operator on single device and SPMD style"""
        if to_dtensor is None:
            to_dtensor = DTensorConverter(self.mesh, args, kwargs)

        rs = func(*args, **kwargs)

        def to_replicate(e: object) -> object:
            return e.full_tensor() if isinstance(e, DTensor) else e

        try:
            # Suppress warnings, this doesn't matter for test_meta.py
            # but it does matter if you want to use this decorator
            # for cross-ref testing, as some tests may be looking at
            # errors
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # for every comb of sharding choices, we test if it works
                for dtensor_args, dtensor_kwargs in to_dtensor:
                    # Only attempt if we managed to convert all tensors to DTensor
                    # (if any of them failed, we're in a mixed tensor situation and
                    # this is not allowed in DTensor)
                    if to_dtensor.successful():
                        # Handle special cases first if there's any
                        # Suppress warnings, this doesn't matter for test_meta.py
                        # but it does matter if you want to use this decorator
                        # for cross-ref testing, as some tests may be looking at
                        # errors
                        dtensor_rs = func(*dtensor_args, **dtensor_kwargs)
                        flat_args = pytree.tree_leaves(dtensor_rs)
                        if any(
                            isinstance(e, torch.Tensor) and e.numel() == 0
                            for e in flat_args
                        ):
                            continue

                        # redistribute/all_gather the results to compare with normal output
                        dtensor_rs = tree_map(to_replicate, dtensor_rs)
                        self.assert_ref_dtensor_equal(dtensor_rs, rs)
                    else:
                        raise RuntimeError(
                            f"failed to convert args to DTensor; "
                            f"originally (*{args}, **{kwargs})"
                        )
        except Exception as e:
            raise RuntimeError(
                f"failed to run: {resolve_name(func)}, with (*{args}, **{kwargs})"
            ) from e

        return rs
