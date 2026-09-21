"""Test that torch.musa device query APIs are constant folded by dynamo
"""

# pylint: disable=missing-function-docstring, unused-import, unnecessary-lambda
import pytest
import torch
from torch._dynamo.utils import common_constant_types
import torch_musa


# Mirrors torch.cuda: upstream constant folds exactly these four.
FOLDED_QUERIES = {
    "is_available": lambda: int(torch.musa.is_available()),
    "current_device": lambda: torch.musa.current_device(),
    "is_initialized": lambda: int(torch.musa.is_initialized()),
    "get_device_properties": lambda: torch.musa.get_device_properties(
        0
    ).multi_processor_count,
}


class TestDeviceQuery:
    """Folded device query APIs must not graph break under fullgraph=True"""

    @pytest.mark.parametrize("name", sorted(FOLDED_QUERIES))
    def test_no_graph_break(self, name):
        query = FOLDED_QUERIES[name]

        def func(x):
            return x + query()

        torch._dynamo.reset()
        x = torch.ones(4, device="musa")
        compiled = torch.compile(func, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), func(x))

    def test_guarded_branch(self):
        def func(x):
            if not torch.musa.is_available():
                return x
            return x + torch.musa.current_device()

        torch._dynamo.reset()
        x = torch.ones(4, device="musa")
        compiled = torch.compile(func, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), func(x))

    def test_props_are_constant(self):
        assert torch_musa._MUSAC._MusaDeviceProperties in common_constant_types
