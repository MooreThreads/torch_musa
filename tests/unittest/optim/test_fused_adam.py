"""Fused Adam and AdamW optimizer tests."""

# pylint: disable=unused-import,invalid-name,not-callable,use-dict-literal,C0116
import os
import functools

import pytest
import torch

from fused_adam_test_utils import (
    amp_training_test,
    basic_test,
    empty_and_missing_grad_test,
    found_inf_test,
    large_numel_indexing_test,
    metadata_flush_tail_after_reset_test,
    metadata_flush_test,
    state_dict_test,
    tail_guard_test,
    tail_test,
    tensor_lr_dtype_test,
    training_test,
    unaligned_test,
)
from torch_musa import testing
from torch_musa.optim import FusedAdam, FusedAdamW


OPTIMIZER_CASES = [
    pytest.param(FusedAdam, torch.optim.Adam, "adam", id="adam"),
    pytest.param(FusedAdamW, torch.optim.AdamW, "adamw", id="adamw"),
    pytest.param(
        functools.partial(torch.optim.Adam, fused=True),
        torch.optim.Adam,
        "torch-adam",
        id="torch-adam",
    ),
    pytest.param(
        functools.partial(torch.optim.AdamW, fused=True),
        torch.optim.AdamW,
        "torch-adamw",
        id="torch-adamw",
    ),
]


class TestFusedAdam:
    """Shared test suite for FusedAdam and FusedAdamW."""

    @pytest.mark.parametrize("optimizer_cls,reference_cls,name", OPTIMIZER_CASES)
    @pytest.mark.skipif(
        os.environ.get("TORCH_MUSA_RUN_LARGE_OPTIMIZER_TESTS") != "1",
        reason="set TORCH_MUSA_RUN_LARGE_OPTIMIZER_TESTS=1 to run the large regression test",
    )
    def test_large_numel_indexing(self, optimizer_cls, reference_cls, name):
        del reference_cls, name
        large_numel_indexing_test(optimizer_cls)

    @pytest.mark.parametrize("optimizer_cls,reference_cls,name", OPTIMIZER_CASES)
    @pytest.mark.skipif(
        os.environ.get("TORCH_MUSA_RUN_LARGE_OPTIMIZER_TESTS") != "1",
        reason="set TORCH_MUSA_RUN_LARGE_OPTIMIZER_TESTS=1 to run the large regression test",
    )
    def test_large_numel_indexing_amsgrad(self, optimizer_cls, reference_cls, name):
        del reference_cls, name
        large_numel_indexing_test(optimizer_cls, amsgrad=True)

    @pytest.mark.parametrize("optimizer_cls,reference_cls,name", OPTIMIZER_CASES)
    @pytest.mark.skipif(
        os.environ.get("TORCH_MUSA_RUN_LARGE_OPTIMIZER_TESTS") != "1",
        reason="set TORCH_MUSA_RUN_LARGE_OPTIMIZER_TESTS=1 to run the metadata flush test",
    )
    def test_metadata_flush_boundary(self, optimizer_cls, reference_cls, name):
        del reference_cls, name
        metadata_flush_test(optimizer_cls)

    @pytest.mark.parametrize("optimizer_cls,reference_cls,name", OPTIMIZER_CASES)
    @pytest.mark.skipif(
        os.environ.get("TORCH_MUSA_RUN_LARGE_OPTIMIZER_TESTS") != "1",
        reason="set TORCH_MUSA_RUN_LARGE_OPTIMIZER_TESTS=1 to run the metadata tail test",
    )
    def test_metadata_flush_tail_after_reset(self, optimizer_cls, reference_cls, name):
        del name
        weight_decay = 0.0 if reference_cls is torch.optim.Adam else 0.1
        metadata_flush_tail_after_reset_test(optimizer_cls, reference_cls, weight_decay)

    @pytest.mark.parametrize("optimizer_cls,reference_cls,name", OPTIMIZER_CASES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("amsgrad", [False, True])
    @pytest.mark.parametrize("weight_decay", [0.0, 0.1])
    def test_tail(
        self,
        optimizer_cls,
        reference_cls,
        name,
        dtype,
        amsgrad,
        weight_decay,
    ):
        del name
        tail_test(optimizer_cls, reference_cls, dtype, weight_decay, amsgrad)

    @pytest.mark.parametrize("optimizer_cls,reference_cls,name", OPTIMIZER_CASES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_tail_guard(self, optimizer_cls, reference_cls, name, dtype):
        del name
        weight_decay = 0.0 if optimizer_cls is FusedAdam else 0.1
        tail_guard_test(optimizer_cls, reference_cls, dtype, weight_decay)

    @pytest.mark.parametrize("optimizer_cls,reference_cls,name", OPTIMIZER_CASES)
    @pytest.mark.parametrize("lr_dtype", [torch.float64])
    def test_tensor_lr_dtype(self, optimizer_cls, reference_cls, name, lr_dtype):
        del name
        tensor_lr_dtype_test(optimizer_cls, reference_cls, lr_dtype)

    @pytest.mark.parametrize("optimizer_cls,reference_cls,name", OPTIMIZER_CASES)
    @pytest.mark.parametrize("found_inf", [0.0, 1.0])
    def test_found_inf(self, optimizer_cls, reference_cls, name, found_inf):
        del reference_cls, name
        found_inf_test(optimizer_cls, found_inf)

    @pytest.mark.parametrize("optimizer_cls,reference_cls,name", OPTIMIZER_CASES)
    def test_empty_and_missing_grad(self, optimizer_cls, reference_cls, name):
        del name
        if reference_cls is torch.optim.Adam:
            weight_decay, expected_active = 0.0, 0.99
        else:
            weight_decay, expected_active = 0.1, 0.989
        empty_and_missing_grad_test(optimizer_cls, weight_decay, expected_active)

    @pytest.mark.parametrize("optimizer_cls,reference_cls,name", OPTIMIZER_CASES)
    @pytest.mark.parametrize("weight_decay", [0.0, 1.0])
    @pytest.mark.parametrize("amsgrad", [False, True])
    def test_basic(self, optimizer_cls, reference_cls, name, weight_decay, amsgrad):
        del name
        basic_test(optimizer_cls, reference_cls, weight_decay, amsgrad, torch.float32)

    @pytest.mark.parametrize("optimizer_cls,reference_cls,name", OPTIMIZER_CASES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("amsgrad", [False, True])
    @pytest.mark.parametrize("weight_decay", [0.0, 0.1])
    def test_unaligned(
        self, optimizer_cls, reference_cls, name, dtype, amsgrad, weight_decay
    ):
        del name
        unaligned_test(optimizer_cls, reference_cls, dtype, amsgrad, weight_decay)

    @pytest.mark.parametrize("optimizer_cls,reference_cls,name", OPTIMIZER_CASES)
    def test_fp32_training(self, optimizer_cls, reference_cls, name):
        del name
        training_test(optimizer_cls, reference_cls)

    @pytest.mark.parametrize("optimizer_cls,reference_cls,name", OPTIMIZER_CASES)
    @pytest.mark.skipif(
        testing.get_musa_arch() < 22, reason="Skipped due to limited support"
    )
    def test_amp_training(self, optimizer_cls, reference_cls, name):
        del name
        amp_training_test(optimizer_cls, reference_cls)

    @pytest.mark.parametrize("optimizer_cls,reference_cls,name", OPTIMIZER_CASES)
    def test_state_dict(self, optimizer_cls, reference_cls, name):
        del reference_cls, name
        state_dict_test(optimizer_cls)
