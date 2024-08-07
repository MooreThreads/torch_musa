import torch

import operator_benchmark as op_bench


"""Microbenchmarks for binary operators."""


# Benchmark ops performance with broadcast
binary_ops_bcast_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["aten2", torch.atan2],
        ["add", torch.add],
        ["sub", torch.sub],
        ["mul", torch.mul],
        ["eq", torch.eq],
        ["ne", torch.ne],
        ["le", torch.le],
        ["gt", torch.gt],
        ["ge", torch.ge],
        ["min", torch.min],
        ["max", torch.max],
    ],
)

# Configs with broadcast
binary_configs_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[64, 1, 64], [1, 64, 1]],
    ],
    cross_product_configs={
        "device": ["musa"],
        "dtype": [torch.float],
    },
    tags=["short"],
)


class BinaryOpBcastBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_one, in_two, dtype, device, op_func):
        self.inputs = {
            "in_one": torch.randn(in_one, device=device).to(dtype=dtype),
            "in_two": torch.randn(in_two, device=device).to(dtype=dtype),
        }
        self.op_func = op_func

    def forward(self, in_one, in_two):
        return self.op_func(in_one, in_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_bcast_list, binary_configs_broadcast, BinaryOpBcastBenchmark
)


def copy(in1, in2):
    return in1.copy_(in2)


# Benchmark ops performance without broadcast
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", torch.add],
        ["copy_", copy],
    ],
)

binary_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["musa"],
        "dtype_one": [torch.int32],
        "dtype_two": [torch.int32],
    },
    tags=["short"],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["musa"],
    dtype_one=[torch.int8, torch.int32],
    dtype_two=[torch.int8, torch.int32],
    tags=["long"],
)


class BinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        self.inputs = {
            "input_one": torch.randn(M, N, K, device=device).to(dtype=dtype_one),
            "input_two": torch.randn(M, N, K, device=device).to(dtype=dtype_two),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, BinaryOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
