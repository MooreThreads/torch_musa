import numpy
import torch
from torch.nn import functional as F

import operator_benchmark as op_bench
from torch_musa import testing


"""
GroupNorm Test
"""


groupnorm_configs_short = op_bench.cross_product_configs(
    dims=((32, 8, 16), (32, 8, 56, 56), (2, 16, 512, 512)),
    num_groups=(2, 4),
    dtype=(torch.float, torch.half, torch.bfloat16),
    device=("musa",),
    tags=["short"],
)


class GroupNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, dims, num_groups, dtype, device):
        num_channels = dims[1]
        self.inputs = {
            "input": (torch.rand(*dims).to(dtype).to(device) - 0.5) * 256,
            "num_groups": num_groups,
            "weight": torch.rand(num_channels, dtype=dtype, device=device),
            "bias": torch.rand(num_channels, dtype=dtype, device=device),
            "eps": 1e-5,
        }

    def forward(self, input, num_groups: int, weight, bias, eps: float):
        return F.group_norm(input, num_groups, weight=weight, bias=bias, eps=eps)


op_bench.generate_pt_test(groupnorm_configs_short, GroupNormBenchmark)


"""
LayerNorm test
"""

layernorm_configs_short = op_bench.cross_product_configs(
    dims=((2, 16, 512, 512)),
    normaliz_begin_dim=(1, 2, 3),
    dtype=(torch.float, torch.half, torch.bfloat16),
    has_bias=(True, False),
    tags=["short"],
)


class LayerNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, dims, normaliz_begin_dim, has_bias, dtype):
        input = (torch.rand(*dims, device="musa", dtype=dtype) - 0.5) * 256
        normalized_shape = dims[normaliz_begin_dim:]
        self.inputs = {
            "input": input,
            "normalized_shape": normalized_shape,
            "weight": torch.rand(normalized_shape, dtype=dtype, device="musa"),
            "bias": (
                torch.rand(normalized_shape, dtype=dtype, device="musa")
                if has_bias
                else None
            ),
            "eps": 1e-5,
        }

    def forward(self, input, normalized_shape, weight, bias, eps: float):
        return F.layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps)


"""
InstanceNorm test
"""

instancenorm_configs_short = op_bench.cross_product_configs(
    dims=((32, 8, 16), (32, 8, 56, 56), (2, 16, 512, 512)),
    dtype=(torch.float, torch.half, torch.bfloat16),
    has_bias=(True, False),
    tags=["short"],
)


class InstanceNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, dims, dtype, has_bias):
        num_channels = dims[1]
        self.inputs = {
            "input": (torch.rand(*dims, device="musa", dtype=dtype) - 0.5) * 256,
            "weight": torch.rand(num_channels, dtype=dtype, device="musa"),
            "bias": torch.rand(num_channels, dtype=dtype, device="musa"),
            "eps": 1e-5,
        }

    def forward(self, input, weight, bias, eps: float):
        return F.instance_norm(input, weight=weight, bias=bias, eps=eps)


op_bench.generate_pt_test(instancenorm_configs_short, InstanceNormBenchmark)

dtypes = testing.get_float_types()

batchnorm1d_configs_short = op_bench.config_list(
    attr_names=["N", "C", "L"],
    attrs=[
        [1, 256, 3136],
        [
            2,
            4,
            512,
        ],
        [2, 1024, 0],
    ],
    cross_product_configs={
        "device": ["musa"],
        "training": [True, False],
        "dtype": dtypes,
    },
    tags=["short"],
)


batchnorm1d_configs_long = op_bench.cross_product_configs(
    N=[2, 4, 8, 16],
    C=[3, 16, 32],
    L=[0, 512, 3136],
    device=["musa"],
    training=[True, False],
    dtype=dtypes,
    tags=["long"],
)


class BatchNorm1DBenchamrk(op_bench.TorchBenchmarkBase):
    def init(self, N, C, L, device, training, dtype):
        # using F.batch_norm
        self.inputs = {}
        if L == 0:
            # input with [N, C]
            self.inputs["input_data"] = torch.rand(
                N, C, device=device, requires_grad=self.auto_set(), dtype=dtype
            )
        else:
            self.inputs["input_data"] = torch.rand(
                N, C, L, device=device, requires_grad=self.auto_set(), dtype=dtype
            )

        other_inputs = {
            "mean": torch.rand(C, device=device, dtype=dtype),
            "var": torch.rand(C, device=device, dtype=dtype),
            "weight": torch.rand(C, device=device, dtype=dtype),
            "bias": torch.rand(C, device=device, dtype=dtype),
            "training": training,
        }

        self.inputs.update(other_inputs)

    def forward(self, inpu_data, mean, var, weight, bias, training):
        return F.batch_norm(inpu_data, mean, var, weight, bias, training)


op_bench.generate_pt_test(
    batchnorm1d_configs_short + batchnorm1d_configs_long, BatchNorm1DBenchamrk
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
