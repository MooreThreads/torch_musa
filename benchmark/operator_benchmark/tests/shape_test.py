import torch

import operator_benchmark as op_bench

from torch_musa import testing


"""
Contiguous Test
"""

nhwc2nchw_contiguous_configs_short = op_bench.cross_product_configs(
    N=[1, 4, 8],
    C=[3, 16],
    HW=[512, 1024],
    device=["musa"],
    tags=["short"],
    dtype=testing.get_float_types(),
)


class NHWC2NCHWContiguousBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, C, HW, device, dtype):
        self.inputs = {
            "x": torch.rand(
                N,
                C,
                HW,
                HW,
                device=device,
                dtype=dtype,
            ).to(memory_format=torch.channels_last)
        }

    def forward(self, x):
        return x.contiguous()


op_bench.generate_pt_test(
    nhwc2nchw_contiguous_configs_short, NHWC2NCHWContiguousBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
