"""Unit test for nn.parallel.scatter"""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
from torch.nn.parallel import scatter
import torch_musa

from torch_musa.testing.common_fsdp import skip_if_lt_x_gpu

input_shapes = [(4, 5), (3, 4)]
devices = [
    [torch.device("musa:0")],
    [torch.device("musa:0"), torch.device("musa:1")],
]

NUM_GPUS_FOR_TESTING_LOSS_PARALLEL = 2

@skip_if_lt_x_gpu(NUM_GPUS_FOR_TESTING_LOSS_PARALLEL)
@pytest.mark.parametrize("input_shape", input_shapes)
@pytest.mark.parametrize("devices", devices)
def test_nn_parallel_scatter(input_shape, devices):
    torch.cuda.device = torch.musa.device
    torch.cuda.current_stream = torch.musa.current_stream

    data = torch.randn(input_shape)

    scatter_outputs = scatter(data, devices)
    results = torch.chunk(data, len(devices))

    for scatter_output, result in zip(scatter_outputs, results):
        torch.allclose(scatter_output.cpu(), result)
