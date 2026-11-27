"""Regression tests for Inductor Triton reduction-loop CSE lifetime."""

# pylint: disable=too-few-public-methods

import pytest
import torch
from torch._dynamo import reset as reset_dynamo
from torch._inductor import config
from torch._inductor.utils import fresh_inductor_cache, run_and_get_kernels
from torch.testing import FileCheck

from torch_musa.testing.base_test_tool import _HAS_TRITON


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
class TestInductorIndexing:
    """Test Inductor indexing code generation on MUSA."""

    def test_indirect_load_reduction(self):
        """Ensure an indirect load is regenerated in each reduction loop."""

        def function(input_tensor, source, index):
            index = index.unsqueeze(-1)
            selected = torch.gather(source, -2, index)
            updated = selected + input_tensor.mean(dim=-1, keepdim=True)
            return input_tensor / updated, source.scatter(-2, index, updated)

        input_tensor = torch.ones(2, 2, 8, device="musa")
        source = torch.arange(
            1, 17, device="musa", dtype=torch.float32
        ).reshape(2, 8, 1)
        index = torch.arange(2, device="musa").repeat(2, 1)

        reset_dynamo()
        with fresh_inductor_cache(), config.patch(  # pylint: disable=no-member
            {
                # MUSA Triton currently cannot link device_assert printf relocations.
                "assert_indirect_indexing": False,
                "force_disable_caches": True,
                "triton.persistent_reductions": False,
            }
        ):
            actual, kernels = run_and_get_kernels(
                torch.compile(function, fullgraph=True, dynamic=True),
                input_tensor,
                source,
                index,
                remove_quote=True,
            )

        expected = function(input_tensor, source, index)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

        reduction_kernels = [kernel for kernel in kernels if "tl.sum(" in kernel]
        assert len(reduction_kernels) == 1

        indirect_load = r"tl\.load\([^\n]*\+ \(tmp\d+"
        (
            FileCheck()
            .check("for r0_offset in tl.range")
            .check_regex(indirect_load)
            .check("tl.store")
            .check_regex(indirect_load)
            .run(reduction_kernels[0])
        )
 