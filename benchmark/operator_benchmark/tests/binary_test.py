import torch
import numpy as np

import operator_benchmark as op_bench


"""Microbenchmarks for binary operators."""

class BinaryBenchmarkBase(op_bench.TorchBenchmarkBase):
    def __init__(self):
        super().__init__()
        self.op_func = None

    def calc_flops(self):
        super().calc_flops()
        if self._is_backward:
            return -1
        assert self.inputs["in_one"] is not None
        assert self.inputs["in_two"] is not None
        input_one_shape = self.inputs["in_one"].shape
        input_two_shape = self.inputs["in_two"].shape

        output_shape = []
        for i in range(min(len(input_one_shape), len(input_two_shape))):
            output_shape.append(max(input_one_shape[-(i + 1)], input_two_shape[-(i + 1)]))
        output_shape = output_shape[::-1]
        flops = np.prod(output_shape)
        return int(flops)

    def calc_memory(self):
        super().calc_memory()
        if self._is_backward:
            return -1
        input_one_shape = self.inputs["in_one"].shape
        input_two_shape = self.inputs["in_two"].shape

        output_shape = []
        for i in range(min(len(input_one_shape), len(input_two_shape))):
            output_shape.append(max(input_one_shape[-(i + 1)], input_two_shape[-(i + 1)]))
        output_shape = output_shape[::-1]

        in_one_dtype = self.inputs["in_one"].dtype
        in_two_dtype = self.inputs["in_two"].dtype

        byte_val = []
        for dtype in [in_one_dtype, in_two_dtype]:
            if dtype in [torch.float16, torch.bfloat16, torch.int16]:
                byte_val.append(2)
            elif dtype in [torch.bool, torch.int8]:
                byte_val.append(1)
            else:
                byte_val.append(4)

        output_dtype = max(byte_val)
        if self.op_func in [torch.eq, torch.ne, torch.le, torch.gt, torch.ge]:
            output_dtype = 1
        byte_val.append(output_dtype)

        memory = (
            byte_val[0] * np.prod(input_one_shape)
            + byte_val[1] * np.prod(input_two_shape)
            + byte_val[2] * np.prod(output_shape)
        )
        return int(memory)


# Benchmark ops performance with broadcast
binary_ops_bcast_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        # ["aten2", torch.atan2],
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
        [[256, 1, 256], [1, 256, 1]],
    ],
    cross_product_configs={
        "device": ["musa"],
        "dtype_one": [torch.float32, torch.float16],
        "dtype_two": [torch.float32, torch.float16],
    },
    tags=["short"],
)


class BinaryOpBcastBenchmark(BinaryBenchmarkBase):
    def init(self, in_one, in_two, dtype_one, dtype_two, device, op_func):
        self.inputs = {
            "in_one": torch.randn(in_one, device=device).to(dtype=dtype_one),
            "in_two": torch.randn(in_two, device=device).to(dtype=dtype_two),
        }
        self.op_func = op_func

    def forward(self, in_one, in_two):
        return self.op_func(in_one, in_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_bcast_list, binary_configs_broadcast, BinaryOpBcastBenchmark
)


# Benchmark ops performance without broadcast
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", torch.add],
    ],
)

binary_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
        [256, 512, 512],
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


class BinaryOpBenchmark(BinaryBenchmarkBase):
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        self.inputs = {
            "in_one": torch.randn(M, N, K, device=device).to(dtype=dtype_one),
            "in_two": torch.randn(M, N, K, device=device).to(dtype=dtype_two),
        }
        self.op_func = op_func

    def forward(self, in_one, in_two):
        return self.op_func(in_one, in_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, BinaryOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
