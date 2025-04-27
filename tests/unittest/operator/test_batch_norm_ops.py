"""This module contains unit tests for batch normalization operations."""

import math
import torch
import pytest
from torch_musa import testing


def batch_norm_gather_stats_with_counts_cpu(
    mean_all, invstd_all, running_mean, running_var, epsilon, momentum, counts
):
    """
    Function to gather statistics with counts for batch normalization on cpu.
    """
    world_size, feature_size = mean_all.shape

    save_mean = torch.empty(feature_size, dtype=mean_all.dtype, device=mean_all.device)
    save_invstd = torch.empty(
        feature_size, dtype=invstd_all.dtype, device=invstd_all.device
    )

    for i in range(feature_size):
        avg = 0.0
        var_n = 0.0
        n = 0.0
        for j in range(world_size):
            count_j = counts[j].item()
            m = mean_all[j, i].item()
            v = 1.0 / invstd_all[j, i].item()
            v = (v * v - epsilon) * count_j
            if n + count_j != 0:
                factor = 1.0 / (n + count_j)
            else:
                factor = 0.0
            var_n += v + (avg - m) * (avg - m) * n * count_j * factor
            avg = n * factor * avg + count_j * factor * m
            n += count_j

        save_mean[i] = avg
        if n > 0:
            sqrt_val = math.sqrt(var_n / n + epsilon)
            save_invstd[i] = 1.0 / sqrt_val
        else:
            save_invstd[i] = 0.0

        if running_mean is not None:
            running_mean[i] = (1 - momentum) * running_mean[i].item() + momentum * avg
        if running_var is not None:
            unbiased_var = var_n / (n - 1) if n > 1 else var_n
            running_var[i] = (1 - momentum) * running_var[
                i
            ].item() + momentum * unbiased_var

    return save_mean, save_invstd


def batch_norm_elemt_cpu(input_data, weight, bias, mean, invstd):
    """
    Apply element-wise batch normalization to the input data on cpu.
    """
    normalized = (input_data - mean[None, :, None, None]) * invstd[None, :, None, None]
    output = weight[None, :, None, None] * normalized + bias[None, :, None, None]

    return output


def batch_norm_backward_reduce_cpu(grad_output, input_data, mean, invstd, weight=None):
    """
    Compute the reduction step for the backward pass of batch normalization on cpu.
    """
    sum_dy = grad_output.sum(dim=(0, 2, 3))
    sum_dy_xmu = (grad_output * (input_data - mean[None, :, None, None])).sum(
        dim=(0, 2, 3)
    )
    if weight is not None:
        grad_weight = sum_dy_xmu * invstd
        grad_bias = sum_dy
    else:
        grad_weight = torch.tensor([])
        grad_bias = torch.tensor([])

    return sum_dy, sum_dy_xmu, grad_weight, grad_bias


def batch_norm_backward_elemt_cpu(
    grad_output: torch.Tensor,
    input_data: torch.Tensor,
    mean: torch.Tensor,
    inv_std: torch.Tensor,
    weight: torch.Tensor,
    sum_dy: torch.Tensor,
    sum_dy_xmu: torch.Tensor,
    count: torch.Tensor,
):
    """
    Compute the element-wise backward pass for batch normalization on cpu.
    """
    _, num_channels, _, _ = input_data.shape
    norm_fct = 1.0 / count.float().sum().item()

    grad_input = torch.empty_like(input_data)

    m_c = mean.view(1, num_channels, 1, 1)
    m_dy_c = sum_dy.view(1, num_channels, 1, 1) * norm_fct
    factor_1_c = inv_std.view(1, num_channels, 1, 1)
    factor_2_c = (
        weight.view(1, num_channels, 1, 1) * factor_1_c
        if weight is not None
        else factor_1_c
    )
    factor_1_c = (
        factor_1_c * factor_1_c * sum_dy_xmu.view(1, num_channels, 1, 1) * norm_fct
    )

    grad_input = (grad_output - m_dy_c - (input_data - m_c) * factor_1_c) * factor_2_c

    return grad_input


all_support_types = [torch.float32]

input_data_batch_norm_gather_stats_with_counts = [
    {
        "input": torch.randn(4, 100, 4, 4),
        "mean_all": torch.randn(4, 100),
        "invstd_all": torch.randn(4, 100),
        "running_mean": torch.randn(100),
        "running_var": torch.randn(100),
        "counts": torch.randint(1, 100, (4,), dtype=torch.int32),
    },
    {
        "input": torch.randn(8, 100, 8, 8),
        "mean_all": torch.randn(8, 100),
        "invstd_all": torch.randn(8, 100),
        "running_mean": torch.randn(100),
        "running_var": torch.randn(100),
        "counts": torch.randint(1, 100, (8,), dtype=torch.int32),
    },
    {
        "input": torch.randn(4, 100, 4, 4).to(memory_format=torch.channels_last),
        "mean_all": torch.randn(4, 100),
        "invstd_all": torch.randn(4, 100),
        "running_mean": torch.randn(100),
        "running_var": torch.randn(100),
        "counts": torch.randint(1, 100, (4,), dtype=torch.int32),
    },
    {
        "input": torch.randn(8, 100, 4, 8).to(memory_format=torch.channels_last),
        "mean_all": torch.randn(8, 100),
        "invstd_all": torch.randn(8, 100),
        "running_mean": torch.randn(100),
        "running_var": torch.randn(100),
        "counts": torch.randint(1, 100, (8,), dtype=torch.int32),
    },
]


@pytest.mark.parametrize("input_data", input_data_batch_norm_gather_stats_with_counts)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("eps", [1e-5])
def test_batch_norm_gather_stats_with_counts(input_data, dtype, eps):
    """Test the batch norm gather stats with counts functionality."""
    input_data_tensor = input_data["input"].to(dtype)
    mean_all = input_data["mean_all"].to(dtype)
    invstd_all = input_data["invstd_all"].to(dtype)
    running_mean = input_data["running_mean"].to(dtype)
    running_var = input_data["running_var"].to(dtype)
    counts = input_data["counts"].to(dtype)
    momentum = 0.1
    mean_musa, invstd_musa = torch.batch_norm_gather_stats_with_counts(
        input_data_tensor.to("musa"),
        mean_all.to("musa"),
        invstd_all.to("musa"),
        running_mean.to("musa"),
        running_var.to("musa"),
        momentum,
        eps,
        counts.to("musa"),
    )
    mean_cpu, invstd_cpu = batch_norm_gather_stats_with_counts_cpu(
        mean_all,
        invstd_all,
        running_mean,
        running_var,
        momentum=0.1,
        epsilon=eps,
        counts=counts,
    )

    assert testing.DefaultComparator(abs_diff=1e-2)(invstd_musa.cpu(), invstd_cpu)
    assert testing.DefaultComparator(abs_diff=1e-2)(mean_musa.cpu(), mean_cpu)


input_data_batch_norm_elemt = [
    {
        "input": torch.randn(4, 100, 4, 4),
        "mean": torch.randn(100),
        "invstd": torch.randn(100),
        "weight": torch.randn(100),
        "bias": torch.randn(100),
    },
    {
        "input": torch.randn(8, 100, 8, 8),
        "mean": torch.randn(100),
        "invstd": torch.randn(100),
        "weight": torch.randn(100),
        "bias": torch.randn(100),
    },
    {
        "input": torch.randn(4, 100, 4, 4).to(memory_format=torch.channels_last),
        "weight": torch.randn(100),
        "bias": torch.randn(100),
        "mean": torch.randn(100),
        "invstd": torch.randn(100),
    },
    {
        "input": torch.randn(8, 100, 8, 8).to(memory_format=torch.channels_last),
        "weight": torch.randn(100),
        "bias": torch.randn(100),
        "mean": torch.randn(100),
        "invstd": torch.randn(100),
    },
]


@pytest.mark.parametrize("input_data", input_data_batch_norm_elemt)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("eps", [1e-5])
def test_batch_norm_elemt(input_data, dtype, eps):
    """Test the element-wise batch normalization operation."""
    input_data_tensor = input_data["input"].to(dtype)
    mean = input_data["mean"].to(dtype)
    invstd = input_data["invstd"].to(dtype)
    weight = input_data["weight"].to(dtype)
    bias = input_data["bias"].to(dtype)

    output_musa = torch.batch_norm_elemt(
        input_data_tensor.to("musa"),
        weight.to("musa"),
        bias.to("musa"),
        mean.to("musa"),
        invstd.to("musa"),
        eps,
    )

    output_cpu = batch_norm_elemt_cpu(input_data_tensor, weight, bias, mean, invstd)

    assert testing.DefaultComparator(abs_diff=1e-2)(output_musa.cpu(), output_cpu)


input_data_batch_norm_backward_reduce = [
    {
        "grad_output": torch.randn(4, 100, 4, 4),
        "input": torch.randn(4, 100, 4, 4),
        "mean": torch.randn(100),
        "invstd": torch.randn(100),
        "weight": torch.randn(100),
    },
    {
        "grad_output": torch.randn(8, 100, 8, 8),
        "input": torch.randn(8, 100, 8, 8),
        "mean": torch.randn(100),
        "invstd": torch.randn(100),
        "weight": torch.randn(100),
    },
    {
        "grad_output": torch.randn(4, 100, 4, 4).to(memory_format=torch.channels_last),
        "input": torch.randn(4, 100, 4, 4).to(memory_format=torch.channels_last),
        "mean": torch.randn(100),
        "invstd": torch.randn(100),
        "weight": torch.randn(100),
    },
    {
        "grad_output": torch.randn(8, 100, 8, 8).to(memory_format=torch.channels_last),
        "input": torch.randn(8, 100, 8, 8).to(memory_format=torch.channels_last),
        "mean": torch.randn(100),
        "invstd": torch.randn(100),
        "weight": torch.randn(100),
    },
]


@pytest.mark.parametrize("input_data", input_data_batch_norm_backward_reduce)
@pytest.mark.parametrize("dtype", all_support_types)
def test_batch_norm_backward_reduce(input_data, dtype):
    """Test the batch norm backward reduce functionality."""
    grad_output = input_data["grad_output"].to(dtype)
    input_data_tensor = input_data["input"].to(dtype)
    mean = input_data["mean"].to(dtype)
    invstd = input_data["invstd"].to(dtype)
    weight = input_data["weight"].to(dtype)

    (
        grad_mean_musa,
        grad_input_partial_musa,
        grad_weight_msua,
        grad_bias_musa,
    ) = torch.batch_norm_backward_reduce(
        grad_output.to("musa"),
        input_data_tensor.to("musa"),
        mean.to("musa"),
        invstd.to("musa"),
        weight.to("musa"),
        input_g=True,
        weight_g=True,
        bias_g=True,
    )

    (
        grad_mean_cpu,
        grad_input_partial_cpu,
        grad_weight_cpu,
        grad_bias_cpu,
    ) = batch_norm_backward_reduce_cpu(
        grad_output, input_data_tensor, mean, invstd, weight
    )

    assert testing.DefaultComparator(abs_diff=1e-2)(
        grad_input_partial_musa.cpu(), grad_input_partial_cpu
    )
    assert testing.DefaultComparator(abs_diff=1e-2)(
        grad_weight_msua.cpu(), grad_weight_cpu
    )
    assert testing.DefaultComparator(abs_diff=1e-2)(grad_bias_musa.cpu(), grad_bias_cpu)
    assert testing.DefaultComparator(abs_diff=1e-2)(grad_mean_musa.cpu(), grad_mean_cpu)


input_data_batch_norm_backward_elemt = [
    {
        "grad_output": torch.randn(4, 100, 4, 4),
        "input": torch.randn(4, 100, 4, 4),
        "mean": torch.randn(100),
        "invstd": torch.randn(100),
        "weight": torch.randn(100),
        "sum_dy": torch.randn(100),
        "sum_dy_xmu": torch.randn(100),
        "count": torch.randint(1, 100, (4,), dtype=torch.int32),
    },
    {
        "grad_output": torch.randn(8, 100, 8, 8),
        "input": torch.randn(8, 100, 8, 8),
        "mean": torch.randn(100),
        "invstd": torch.randn(100),
        "weight": torch.randn(100),
        "sum_dy": torch.randn(100),
        "sum_dy_xmu": torch.randn(100),
        "count": torch.randint(1, 100, (8,), dtype=torch.int32),
    },
    {
        "grad_output": torch.randn(4, 100, 4, 4).to(memory_format=torch.channels_last),
        "input": torch.randn(4, 100, 4, 4).to(memory_format=torch.channels_last),
        "mean": torch.randn(100),
        "invstd": torch.randn(100),
        "weight": torch.randn(100),
        "sum_dy": torch.randn(100),
        "sum_dy_xmu": torch.randn(100),
        "count": torch.randint(1, 100, (4,), dtype=torch.int32),
    },
    {
        "grad_output": torch.randn(8, 100, 8, 8).to(memory_format=torch.channels_last),
        "input": torch.randn(8, 100, 8, 8).to(memory_format=torch.channels_last),
        "mean": torch.randn(100),
        "invstd": torch.randn(100),
        "weight": torch.randn(100),
        "sum_dy": torch.randn(100),
        "sum_dy_xmu": torch.randn(100),
        "count": torch.randint(1, 100, (8,), dtype=torch.int32),
    },
]


@pytest.mark.parametrize("input_data", input_data_batch_norm_backward_elemt)
@pytest.mark.parametrize("dtype", all_support_types)
def test_batch_norm_backward_elemt(input_data, dtype):
    """Test the element-wise batch normalization backward pass."""
    grad_output = input_data["grad_output"].to(dtype)
    input_data_tensor = input_data["input"].to(dtype)
    mean = input_data["mean"].to(dtype)
    invstd = input_data["invstd"].to(dtype)
    weight = input_data["weight"].to(dtype)
    sum_dy = input_data["sum_dy"].to(dtype)
    sum_dy_xmu = input_data["sum_dy_xmu"].to(dtype)
    count = input_data["count"]

    grad_input_musa = torch.batch_norm_backward_elemt(
        grad_output.to("musa"),
        input_data_tensor.to("musa"),
        mean.to("musa"),
        invstd.to("musa"),
        weight.to("musa"),
        sum_dy.to("musa"),
        sum_dy_xmu.to("musa"),
        count.to("musa"),
    )

    grad_input_cpu = batch_norm_backward_elemt_cpu(
        grad_output, input_data_tensor, mean, invstd, weight, sum_dy, sum_dy_xmu, count
    )

    assert testing.DefaultComparator(abs_diff=1e-2)(
        grad_input_musa.cpu(), grad_input_cpu
    )
