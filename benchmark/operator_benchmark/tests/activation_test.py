import torch
import numpy as np
import operator_benchmark as op_bench

from torch_musa import testing

"""
GELU benchmark
"""
gelu_configs_short = op_bench.cross_product_configs(
    dtype=[torch.float32, torch.half],
    N=[1, 8],
    C=[3],
    H=[512, 1024, 2048],
    W=[512, 1024, 2048],
    device=["musa"],
    tags=["short"],
)


def activation_flops(ratio, input_shape):
    return int(ratio * np.prod(input_shape))


def activation_memory(dtype, input_shape, inplace=False):
    if dtype in [torch.float16, torch.bfloat16]:
        byte_val = 2
    else:
        byte_val = 4
    memory = byte_val * np.prod(input_shape) * (1 if inplace else 2)
    return int(memory)


class GeluBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, N, C, H, W, device, dtype):
        self.inputs = {"input": torch.rand(N, C, H, W, device=device, dtype=dtype)}
    
    def calc_flops(self):
        super().calc_flops()
        if self._is_backward:
            return -1
        return activation_flops(9, self.inputs["input"].shape)
    
    def calc_memory(self):
        super().calc_memory()
        if self._is_backward:
            return -1
        return activation_memory(self.inputs["input"].dtype, self.inputs["input"].shape)

    def forward(self, input):
        return torch.nn.functional.gelu(input)


op_bench.generate_pt_test(gelu_configs_short, GeluBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
