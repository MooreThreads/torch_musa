import torch

import operator_benchmark as op_bench

from torch_musa import testing

"""
GELU benchmark
"""
gelu_configs_short = op_bench.cross_product_configs(
    dtype=testing.get_float_types(),
    N=[1, 4],
    C=[3],
    H=[512, 1024],
    W=[512, 1024],
    device=["musa"],
    tags=["short"],
)


class GeluBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, C, H, W, device, dtype):
        self.inputs = {"input": torch.rand(N, C, H, W, device=device, dtype=dtype)}

    def forward(self, input):
        return torch.nn.functional.gelu(input)

op_bench.generate_pt_test(gelu_configs_short, GeluBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
