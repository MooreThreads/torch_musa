import torch
import torch.nn as nn

import operator_benchmark as op_bench

from torch_musa import testing


# Configs for softmax ops
softmax_configs_short = op_bench.config_list(
    attr_names=["N", "C", "H", "W"],
    attrs=[
        [1, 3, 256, 256],
        [4, 3, 256, 256],
    ],
    cross_product_configs={"device": ["musa"], "dtype": testing.get_float_types()},
    tags=["short"],
)


softmax_configs_long = op_bench.cross_product_configs(
    N=[8, 16],
    C=[3],
    H=[256, 512],
    W=[256, 512],
    device=["musa"],
    dtype=testing.get_float_types(),
    tags=["long"],
)

softmax_configs_llm = op_bench.config_list(
    attr_names=["Q_len", "KV_len"],
    attrs=[[1024, 1024], [32 * 1024, 32 * 1024]],
    cross_product_configs={"device": ["musa"], "dtype": testing.get_float_types()},
    tags="short",
)

softmax_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["Softmax", nn.Softmax],
        ["Softmax2d", nn.Softmax2d],
        ["LogSoftmax", nn.LogSoftmax],
    ],
)

softmax_two_dims_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["LogSoftmax", nn.LogSoftmax],
        ["Softmax", nn.Softmax],
    ],
)

softmax_two_dims_configs = op_bench.config_list(
    attr_names=["N", "seq_len", "dim"],
    attrs=[[700, 23258, 0], [700, 23258, 1], [1024, 23258, 1]],
    cross_product_configs={"device": ["musa"], "dtype": testing.get_float_types()},
    tags=["long"],
)


class SoftmaxBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, C, H, W, device, op_func, dtype):
        self.inputs = {"input": torch.rand(N, C, H, W, device=device, dtype=dtype)}
        self.op_func = op_func()

    def forward(self, input):
        return self.op_func(input)


class Softmax2DimsBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, seq_len, dim, device, op_func, dtype):
        self.inputs = {"input": torch.rand(N, seq_len, device=device, dtype=dtype)}
        self.op_func = op_func(dim=dim)

    def forward(self, input):
        return self.op_func(input)


class SoftmaxLLMBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, Q_len, KV_len, device, dtype):
        self.inputs = {"input": torch.rand(Q_len, KV_len, device=device, dtype=dtype)}
        self.op_func = nn.Softmax(dim=-1)

    def forward(self, input):
        return self.op_func(input)


op_bench.generate_pt_tests_from_op_list(
    softmax_ops_list, softmax_configs_short + softmax_configs_long, SoftmaxBenchmark
)

op_bench.generate_pt_tests_from_op_list(
    softmax_two_dims_ops_list, softmax_two_dims_configs, Softmax2DimsBenchmark
)

op_bench.generate_pt_test(softmax_configs_llm, SoftmaxLLMBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
