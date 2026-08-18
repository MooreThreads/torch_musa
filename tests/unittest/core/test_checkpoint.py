"""Test torch.utils.checkpoint interoperation with the musa backend."""

# pylint: disable=unused-import
import pytest
import torch
from torch.utils.checkpoint import checkpoint

import torch_musa
from torch_musa import testing


def _noisy_mul_under_checkpoint(use_reentrant):
    """Run `x * randn_like(x)` inside a checkpoint, recording what each pass drew."""
    draws = []

    def noisy_mul(tensor):
        noise = torch.randn_like(tensor)
        draws.append(noise.detach().clone())
        return tensor * noise

    x = torch.randn(256, device="musa", requires_grad=True)
    out = checkpoint(
        noisy_mul,
        x,
        use_reentrant=use_reentrant,
        preserve_rng_state=True,
    )
    out.sum().backward()
    return draws, x.grad


@pytest.mark.parametrize("use_reentrant", [False, True])
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_checkpoint_preserves_device_rng(use_reentrant):
    """The backward recompute must replay the forward's random draws bitwise.

    That is the whole contract of `preserve_rng_state=True`.  It only holds if
    `torch.musa._initialized` reads truthy once MUSA has been initialized, because
    `torch.utils.checkpoint` gates the device-RNG save/restore on that attribute.
    """
    draws, grad = _noisy_mul_under_checkpoint(use_reentrant)

    assert len(draws) == 2, f"expected forward + 1 recompute, saw {len(draws)}"
    assert torch.equal(draws[0], draws[1]), "recompute drew different noise"
    # d(x * n)/dx == n, so the gradient must be exactly the noise the forward drew.
    assert torch.equal(grad, draws[0])


def test_checkpoint_dropout_gradient_matches_eager():
    """Same contract for the canonical case: dropout inside a checkpointed block."""
    # nn.Linear initializes its parameters on the CPU, so seed the CPU generator
    # here; the musa generator is what the dropout masks below are drawn from.
    torch.manual_seed(0)
    block = torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.Dropout(0.5), torch.nn.Linear(64, 64)
    ).to("musa")
    block.train()
    x = torch.randn(16, 64, device="musa", requires_grad=True)

    torch.musa.manual_seed(1234)
    block(x).sum().backward()
    reference = [p.grad.detach().clone() for p in block.parameters()]
    reference_input_grad = x.grad.detach().clone()

    block.zero_grad(set_to_none=True)
    x.grad = None
    torch.musa.manual_seed(1234)
    out = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
    out.sum().backward()

    # The dropout mask must be replayed bitwise, but these gradients flow through
    # matmul, so compare them tightly rather than exactly.  Without the mask replay
    # the difference is O(1), several orders of magnitude above this tolerance.
    for expected, param in zip(reference, block.parameters()):
        torch.testing.assert_close(expected, param.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reference_input_grad, x.grad, rtol=1e-5, atol=1e-6)
