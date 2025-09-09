import operator_benchmark as op_bench
import torch
import numpy as np

"""Microbenchmarks for MatMul operator"""


def matmul_flops(inp1_shape, output_shape):
    # matmul flops: 
    flops = 2 * np.prod(output_shape, dtype=np.int64) * inp1_shape[-1]
    return int(flops)


# Configs for PT Matmul operator
mm_short_configs = op_bench.config_list(
    attr_names=["IN", "OC"],
    attrs=[
        [[256, 512], 512],
        [[2, 2048, 4096], 4096],
        [[8, 4096, 4096], 4096],
        [[8, 4096, 4096], 11008],
    ],
    cross_product_configs={
        "dtype": [torch.float32, torch.float16],
        "device": ["musa"],
    },
    tags=["short"],
)


mm_long_configs = op_bench.cross_product_configs(
    IN=[[128, 256], [2, 1024, 2048]],
    OC=[512, 2048, 4096],
    dtype=[torch.float32, torch.float16],
    device=["musa"],
    tags=["long"],
)


class MatMulBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, IN, OC, dtype, device):
        self.inputs = {
            "input_one": (
                torch.rand(IN, dtype=dtype, device=device, requires_grad=self.auto_set())
            ),
            "input_two": (
                torch.rand([IN[-1], OC], dtype=dtype, device=device, requires_grad=self.auto_set())
            ),
        }
        self.set_module_name("matmul")

    def forward(self, input_one, input_two):
        return torch.matmul(input_one, input_two)

    def calc_flops(self):
        super().calc_flops()
        inp1_shape = self.inputs["input_one"].shape
        inp2_shape = self.inputs["input_two"].shape
        output_shape = self.outputs.shape
        if self._is_backward:
            grad_inp1_flops = matmul_flops(output_shape, inp1_shape)
            grad_inp2_flops = matmul_flops(output_shape, inp2_shape)
            return grad_inp1_flops + grad_inp2_flops
        else:
            return matmul_flops(inp1_shape, output_shape)


    def calc_memory(self):
        super().calc_memory()
        inp1_shape = self.inputs["input_one"].shape
        inp2_shape = self.inputs["input_two"].shape
        output_shape = self.outputs.shape
        dtype = self.outputs.dtype
        if dtype in [torch.float16, torch.bfloat16]:
            byte_val = 2
        else:
            byte_val = 4
        memory = byte_val * (
            np.prod(inp1_shape) + np.prod(inp2_shape) + np.prod(output_shape)
        )
        if self._is_backward:
            memory = memory * 2 - byte_val * np.prod(output_shape)

        return int(memory)


total_configs = mm_long_configs + mm_short_configs
op_bench.generate_pt_test(total_configs, MatMulBenchmark)
op_bench.generate_pt_gradient_test(total_configs, MatMulBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
