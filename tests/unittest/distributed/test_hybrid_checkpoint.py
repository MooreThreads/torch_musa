"""Tests for MUSA hybrid checkpointing."""

from collections import defaultdict
from copy import deepcopy

import pytest
import torch
from torch.utils.checkpoint import SelectiveCheckpointContext

from torch_musa.distributed import hybrid_checkpoint as hybrid_ac
from torch_musa.distributed.hybrid_checkpoint import (
    HybridCheckpointConfig,
    apply_hybrid_checkpoint,
)


class _ToyBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim, dim) / dim**0.5)

    def forward(self, value):
        return torch.mm(value, self.weight).relu()


@pytest.fixture(autouse=True)
def _reset_hybrid_handler():
    hybrid_ac._HYBRID_OFFLOAD_HANDLER = None
    yield
    hybrid_ac._HYBRID_OFFLOAD_HANDLER = None


def test_config_validation():
    """Validate rejected Hybrid Checkpoint configurations."""
    matmul_op = torch.ops.aten.mm.default

    with pytest.raises(ValueError, match="num_layers"):
        HybridCheckpointConfig(num_layers=-1)
    with pytest.raises(TypeError, match="OpOverload"):
        HybridCheckpointConfig(save_list=[torch.ops.aten.mm])
    with pytest.raises(ValueError, match="subset"):
        HybridCheckpointConfig(save_list=(), cpu_save_list=(matmul_op,))
    with pytest.raises(ValueError, match="positive integers"):
        HybridCheckpointConfig(
            save_list=(matmul_op,),
            cpu_save_list=(),
            save_frequencies={matmul_op: 0},
        )


def test_save_frequency_policy():
    """Check occurrence counting for forward and recompute independently."""
    matmul_op = torch.ops.aten.mm.default
    linear = torch.ops.aten.linear.default
    policy = hybrid_ac._get_hybrid_policy(
        save_ops={matmul_op, linear},
        offload_ops={matmul_op, linear},
        save_frequencies={matmul_op: 2, linear: 2},
    )
    forward_context = SelectiveCheckpointContext(is_recompute=False)
    recompute_context = SelectiveCheckpointContext(is_recompute=True)

    expected = (
        hybrid_ac._HybridCheckpointPolicy.PREFER_RECOMPUTE,
        hybrid_ac._HybridCheckpointPolicy.PREFER_RECOMPUTE,
        hybrid_ac._HybridCheckpointPolicy.MUST_SAVE_OFFLOAD,
        hybrid_ac._HybridCheckpointPolicy.MUST_SAVE_OFFLOAD,
    )
    ops = (matmul_op, linear, matmul_op, linear)
    forward_policies = tuple(policy(forward_context, op) for op in ops)
    recompute_policies = tuple(policy(recompute_context, op) for op in ops)

    assert forward_policies == expected
    assert recompute_policies == expected


@pytest.mark.parametrize(
    "op",
    (torch.ops.aten.addmm.out, torch.ops.aten.mm.out),
    ids=("addmm.out", "mm.out"),
)
def test_out_variant_cache_replay(op):
    """Cache out variants independently and restore the replay destination."""

    def policy_fn(*_args, **_kwargs):
        return hybrid_ac._HybridCheckpointPolicy.MUST_SAVE

    storage = defaultdict(list)
    caching_mode = hybrid_ac._HybridCachingTorchDispatchMode(
        policy_fn, storage, [], [], enable_offload=False
    )
    cached_mode = hybrid_ac._HybridCachedTorchDispatchMode(
        policy_fn, storage, allow_cache_entry_mutation=False
    )
    lhs = torch.tensor([[1.0, 2.0]])
    replay_lhs = torch.zeros_like(lhs)
    rhs = torch.tensor([[3.0], [4.0]])
    bias = torch.tensor([[5.0]])
    forward_args = (bias, lhs, rhs) if op == torch.ops.aten.addmm.out else (lhs, rhs)
    replay_args = (
        (bias, replay_lhs, rhs) if op == torch.ops.aten.addmm.out else (replay_lhs, rhs)
    )
    forward_out = torch.empty(1, 1)

    with caching_mode:
        returned = op(*forward_args, out=forward_out)
    assert returned is forward_out
    expected = forward_out.clone()

    # Simulate compiled buffer reuse after the saved out-variant call.
    forward_out.fill_(-1)
    replay_out = torch.empty_like(forward_out)
    with cached_mode:
        returned = op(*replay_args, out=replay_out)

    assert returned is replay_out
    torch.testing.assert_close(replay_out, expected)


def test_container_api_cpu_parity():
    """Check CPU parity and the configured container layer limit."""
    torch.manual_seed(7)
    eager_model = torch.nn.Sequential(_ToyBlock(16), _ToyBlock(16))
    checkpoint_model = deepcopy(eager_model)
    config = HybridCheckpointConfig(
        num_layers=1,
        save_list=(torch.ops.aten.mm.default,),
        cpu_save_list=(),
        save_frequencies={torch.ops.aten.mm.default: 1},
    )

    apply_hybrid_checkpoint(checkpoint_model, config)
    assert isinstance(checkpoint_model[0], hybrid_ac._HybridCheckpointWrapper)
    assert isinstance(checkpoint_model[1], _ToyBlock)

    eager_input = torch.randn(4, 16, requires_grad=True)
    checkpoint_input = eager_input.detach().clone().requires_grad_(True)
    eager_output = eager_model(eager_input)
    checkpoint_output = checkpoint_model(checkpoint_input)
    eager_output.sum().backward()
    checkpoint_output.sum().backward()

    torch.testing.assert_close(checkpoint_output, eager_output)
    torch.testing.assert_close(checkpoint_input.grad, eager_input.grad)
    for checkpoint_param, eager_param in zip(
        checkpoint_model.parameters(), eager_model.parameters()
    ):
        torch.testing.assert_close(checkpoint_param.grad, eager_param.grad)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA is not available")
def test_musa_offload_parity_and_reuse():
    """Check MUSA parity and pinned bucket reuse across training steps."""
    torch.manual_seed(11)
    eager_model = torch.nn.Sequential(_ToyBlock(16), _ToyBlock(16))
    eager_model.to("musa")
    checkpoint_model = deepcopy(eager_model)
    config = HybridCheckpointConfig(
        save_list=(torch.ops.aten.mm.default,),
        cpu_save_list=(torch.ops.aten.mm.default,),
        offload_bucket_bytes=(1024,),
    )
    apply_hybrid_checkpoint(checkpoint_model, config)

    eager_input = torch.randn(8, 16, device="musa", requires_grad=True)
    checkpoint_input = eager_input.detach().clone().requires_grad_(True)
    eager_output = eager_model(eager_input)
    checkpoint_output = checkpoint_model(checkpoint_input)
    eager_output.sum().backward()
    checkpoint_output.sum().backward()
    torch.musa.synchronize()

    torch.testing.assert_close(checkpoint_output.cpu(), eager_output.cpu())
    torch.testing.assert_close(checkpoint_input.grad.cpu(), eager_input.grad.cpu())
    for checkpoint_param, eager_param in zip(
        checkpoint_model.parameters(), eager_model.parameters()
    ):
        torch.testing.assert_close(checkpoint_param.grad.cpu(), eager_param.grad.cpu())

    handler = hybrid_ac._get_hybrid_offload_handler()
    assert handler.completed_steps == 1
    assert handler.bucket_allocations > 0
    assert handler.current_layer == 0
    assert not handler.storages

    checkpoint_model.zero_grad(set_to_none=True)
    second_input = torch.randn(8, 16, device="musa", requires_grad=True)
    checkpoint_model(second_input).sum().backward()
    torch.musa.synchronize()

    assert handler.completed_steps == 2
    assert handler.bucket_hits > 0
