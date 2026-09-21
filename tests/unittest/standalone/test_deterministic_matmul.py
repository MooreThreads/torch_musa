"""Test deterministic mode for muDNN xmma operators."""

from contextlib import contextmanager
import pytest

import torch
import torch_musa  # pylint: disable=unused-import
from torch_musa import testing

REPEAT = 10


pytestmark = pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="Deterministic tests are only enabled after mp_21",
)


@contextmanager
def deterministic_algorithms():
    """Temporarily enable deterministic algorithms."""
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def assert_deterministic_result(op):
    """Run an op repeatedly under deterministic mode and compare output bits."""
    with deterministic_algorithms():
        outputs = [op() for _ in range(REPEAT)]

    expected_bits = outputs[0].detach().cpu().contiguous().view(torch.uint8).flatten()
    for output in outputs[1:]:
        output_bits = output.detach().cpu().contiguous().view(torch.uint8).flatten()
        assert output_bits.numel() == expected_bits.numel()
        assert torch.equal(output_bits, expected_bits)


def test_deterministic_matmul():
    torch.manual_seed(0)
    lhs_cpu = torch.randn(96, 20480, dtype=torch.float16)
    rhs_cpu = torch.randn(20480, 1024, dtype=torch.float16)
    lhs = lhs_cpu.musa()
    rhs = rhs_cpu.musa()

    assert_deterministic_result(lambda: torch.matmul(lhs, rhs))


def test_deterministic_batchmatmul():
    torch.manual_seed(1)
    lhs_cpu = torch.randn(1, 96, 20480, dtype=torch.float16)
    rhs_cpu = torch.randn(1, 20480, 1024, dtype=torch.float16)
    lhs = lhs_cpu.musa()
    rhs = rhs_cpu.musa()

    assert_deterministic_result(lambda: torch.bmm(lhs, rhs))


def test_deterministic_groupmatmul():
    torch.manual_seed(0)
    lhs_cpu = torch.randn(1, 2048, dtype=torch.float16)
    rhs_cpu = torch.randn(2048, 512, dtype=torch.float16)
    offs_cpu = torch.arange(256, 2049, 256, dtype=torch.int32)
    lhs = lhs_cpu.musa()
    rhs = rhs_cpu.musa()
    offs = offs_cpu.musa()

    assert_deterministic_result(lambda: torch._grouped_mm(lhs, rhs, offs, None, None))
