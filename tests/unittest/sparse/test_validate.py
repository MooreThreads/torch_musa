"""Test embedding operators."""

# pylint: disable=missing-function-docstring, global-variable-not-assigned, redefined-outer-name, unused-import
import random
import numpy as np
import torch
import pytest

from torch_musa import testing

n = random.randint(1, 128)
m = random.randint(1, 128)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_validate_compressed_sparse_indices():
    global m, n

    input_tensor = torch.randint(low=1, high=n, size=(m, n)).musa()
    crow_indices = [0]
    values = []
    col_indices = []
    index = 0
    for i in range(m):
        for j in range(n):
            if input_tensor[i][j] != 0:
                index += 1
                values.append(input_tensor[i][j])
                col_indices.append(j)
        crow_indices.append(index)
    crow_indices = torch.tensor(crow_indices).musa()
    col_indices = torch.tensor(col_indices).musa()
    values = torch.tensor(values).musa()
    nnz = values.numel()
    nrows = m
    torch._validate_compressed_sparse_indices(
        True, crow_indices, col_indices, nrows, 0, nnz
    )

    # test specific matrtix
    # [[1, 0, 2, 0],
    #  [0, 0, 3, 0],
    #  [4, 5, 0, 0]]
    crow_indices = torch.tensor([0, 2, 3, 5], dtype=torch.int64).musa()
    col_indices = torch.tensor([0, 2, 2, 0, 1], dtype=torch.int64).musa()
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).musa()

    nrows = 3
    nnz = values.numel()

    torch._validate_compressed_sparse_indices(
        True, crow_indices, col_indices, nrows, 0, nnz
    )
