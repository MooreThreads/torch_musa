import operator_benchmark as op_bench
import torch
import torch.nn as nn
import numpy as np

from tests import configs


"""Microbenchmarks for Linear operator."""

def linear_flops(input_shape, weight_shape, bias):
    # linear flops: 2 * batch/batch*len * in_feature * out_feature
    output_shape = list(input_shape[:-1]) + [weight_shape[0]]
    flops = 2 * np.prod(input_shape, dtype=np.int64) * weight_shape[0]
    if not bias:
        flops -= np.prod(output_shape)
    return int(flops)


class LinearBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, IN, OC, BIAS, device, dtype):
        self.inputs = {"input_one": torch.rand(IN, device=device)}
        self.linear = nn.Linear(IN[-1], OC, bias=BIAS).to(device=device)
        self.set_module_name("linear")

    def calc_flops(self):
        super().calc_flops()
        input_shape = self.inputs["input_one"].shape
        weight_shape = self.linear.weight.shape
        output_shape = list(input_shape[:-1]) + [weight_shape[0]]
        bias = self.linear.bias is not None
        if self._is_backward:
            grad_input_flops = linear_flops(output_shape, weight_shape[::-1], False)
            grad_weight_flops = linear_flops(output_shape, input_shape[1:][::-1], False)
            grad_bias_flops = int(np.prod(output_shape)) if bias else 0
            return grad_input_flops + grad_weight_flops + grad_bias_flops
        else:
            return linear_flops(input_shape, weight_shape, bias)

    def calc_memory(self):
        super().calc_memory()
        input_shape = self.inputs["input_one"].shape
        weight_shape = self.linear.weight.shape
        output_shape = self.outputs[0].shape
        if self.linear.weight.dtype in [torch.float16, torch.bfloat16]:
            byte_val = 2
        else:
            byte_val = 4
        memory = byte_val * (
            np.prod(input_shape)
            + np.prod(weight_shape)
            + weight_shape[0]
            + np.prod(output_shape)
        )
        if self._is_backward:
            memory = memory * 2 - byte_val * np.prod(output_shape)

        if self.linear.bias is None:
            memory = memory - byte_val * weight_shape[0]
            if self._is_backward:
                memory = memory - byte_val * weight_shape[0]
    
        return int(memory)

    def forward(self, input_one):
        return self.linear(input_one)


op_bench.generate_pt_test(
    configs.linear_configs_short + configs.linear_configs_long, LinearBenchmark
)
op_bench.generate_pt_gradient_test(
    configs.linear_configs_short + configs.linear_configs_long, LinearBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
