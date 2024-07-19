"""Test index_add operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa
from torch_musa import testing

# Note: muDNN doesn't support float64 or bool for this operator.
# We should enable these two types after fill is implemented with MUSA.
data_type = testing.get_all_support_types()


# copy from https://github.com/pytorch/pytorch/blob/v2.0.0/test/test_mps.py#L5673
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("device", ["musa"])
def test_index_add(device):
    def helper(
        shape,
        dim,
        index,
        source_shape,
        alpha,
        x_dtype=torch.float32,
        idx_dtype=torch.int32,
    ):
        cpu_x = torch.randn(shape, device="cpu", dtype=x_dtype, requires_grad=False)
        x = cpu_x.detach().clone().to(device)

        cpu_idx = torch.tensor(index, device="cpu", dtype=idx_dtype)
        idx = cpu_idx.detach().clone().to(device)

        cpu_source = torch.randn(
            source_shape, device="cpu", dtype=x_dtype, requires_grad=False
        )
        source = cpu_source.detach().clone().to(device)

        idx_result = torch.index_add(x, dim=dim, index=idx, source=source, alpha=alpha)
        idx_result_cpu = torch.index_add(
            cpu_x, dim=dim, index=cpu_idx, source=cpu_source, alpha=alpha
        )
        testing.DefaultComparator()(idx_result, idx_result_cpu)

        idx_result = torch.rand_like(idx_result)
        idx_result_cpu = torch.rand_like(idx_result_cpu)
        torch.index_add(
            x, dim=dim, index=idx, source=source, alpha=alpha, out=idx_result
        )
        torch.index_add(
            cpu_x,
            dim=dim,
            index=cpu_idx,
            source=cpu_source,
            alpha=alpha,
            out=idx_result_cpu,
        )
        testing.DefaultComparator()(idx_result, idx_result_cpu)

        idx_result = x.index_add(dim=dim, index=idx, source=source, alpha=alpha)
        idx_result_cpu = cpu_x.index_add(
            dim=dim, index=cpu_idx, source=cpu_source, alpha=alpha
        )
        testing.DefaultComparator()(idx_result, idx_result_cpu)

        x.index_add_(dim=dim, index=idx, source=source, alpha=alpha)
        cpu_x.index_add_(dim=dim, index=cpu_idx, source=cpu_source, alpha=alpha)
        testing.DefaultComparator()(x, cpu_x)

    helper((2, 8, 4, 5), 0, [0, 1, 0], (3, 8, 4, 5), 5)
    helper((8, 8, 4, 5), 0, [7], (1, 8, 4, 5), 6.0)
    helper((2, 8, 4, 5), 1, [0, 3, 7], (2, 3, 4, 5), 5)
    helper((2, 8, 4, 5), 2, [3, 0], (2, 8, 2, 5), 3.0)
    helper((2, 8, 4, 5), 3, [2, 3, 0], (2, 8, 4, 3), 4)
    helper((2, 3, 3), -1, [1, 2], (2, 3, 2), 6.0)
    # test result dim=1
    helper((2,), 0, [1], (1,), 6.0)
    helper(2, 0, 1, 1, 6)
    # test float16
    helper((2,), 0, [1], (1,), 6.0, x_dtype=torch.float16)


# copy from https://github.com/pytorch/pytorch/blob/v2.0.0/test/test_torch.py#L3966
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("device", ["musa"])
def test_index_add_special(device):
    shape = (0, 1, 2, 0)
    x = torch.randn(shape, device=device)
    c = x.clone()
    c_clone = c.clone()
    ind_empty = torch.tensor([], dtype=torch.int64, device=device)
    ind_01 = torch.tensor([0, 1], dtype=torch.int64, device=device)

    testing.DefaultComparator()(
        c_clone, c.index_add_(0, ind_empty, torch.empty((0, 1, 2, 0), device=device))
    )
    testing.DefaultComparator()(
        c_clone, c.index_add_(2, ind_empty, torch.empty((0, 1, 0, 0), device=device))
    )
    testing.DefaultComparator()(
        c_clone, c.index_add_(2, ind_01, torch.empty((0, 1, 2, 0), device=device))
    )
