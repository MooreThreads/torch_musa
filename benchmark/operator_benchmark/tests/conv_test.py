import operator_benchmark as op_bench
import torch
import torch.nn as nn
import numpy as np

from tests import configs

"""
Microbenchmarks for Conv1d and ConvTranspose1d operators.
"""
def linear_flops(input_shape, weight_shape, bias):
    # linear flops: 2 * batch/batch*len * in_feature * out_feature
    output_shape = list(input_shape[:-1]) + [weight_shape[0]]
    flops = 2 * np.prod(input_shape, dtype=np.int64) * weight_shape[0]
    if not bias:
        flops -= np.prod(output_shape)
    return int(flops)

class ConvBenchmarkBase(op_bench.TorchBenchmarkBase):
    def __init__(self):
        super().__init__()
        self.conv = None

    def calc_flops(self):
        super().calc_flops()
        assert self.outputs is not None
        assert self.conv is not None
        input_shape = self.inputs["input"].shape
        N, IC = input_shape[:2]
        kernel_shape = self.conv.weight.shape
        KK = np.prod(kernel_shape[2:])
        output_shape = self.outputs.shape
        OC = output_shape[1]
        OHOW = np.prod(output_shape[2:])
        groups = self.conv.groups
        bias = self.conv.bias is not None
        if self._is_backward:
            # grad_input_flops is a little smaller
            grad_input_flops = linear_flops([N, KK*IC, OC], [OHOW, OC], False)
            grad_weight_flops = linear_flops([N, OC, OHOW], [KK*IC, OHOW], False)
            grad_bias_flops = int(np.prod(output_shape)) if bias else 0
            flops = grad_input_flops + grad_weight_flops
            flops /= groups
            flops += grad_bias_flops
            return int(flops)
        else:
            flops = 2 * np.prod(output_shape, dtype=np.int64) * KK * IC
            flops /= groups
            if self.conv.bias is None:
                flops = flops - np.prod(output_shape, dtype=np.int64)
            return int(flops)

    def calc_memory(self):
        super().calc_memory()
        input_shape = self.inputs["input"].shape
        weight_shape = self.conv.weight.shape
        output_shape = self.outputs.shape
        if self.conv.weight.dtype in [torch.float16, torch.bfloat16]:
            byte_val = 2
        else:
            byte_val = 4

        memory = (
            byte_val * np.prod(input_shape)
            + byte_val * np.prod(weight_shape)
            + byte_val * weight_shape[0]
            + byte_val * np.prod(output_shape) / np.prod(self.conv.stride)
        )
        if self._is_backward:
            memory = memory * 2 - byte_val * np.prod(output_shape)

        if self.conv.bias is None:
            memory = memory - byte_val * weight_shape[0]
            if self._is_backward:
                memory = memory - byte_val * weight_shape[0]

        return int(memory)


class Conv1dBenchmark(ConvBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, L, device, dtype, memory_format):
        self.inputs = {
            "input": torch.rand(N, IC, L, device=device, dtype=dtype, requires_grad=self.auto_set())
        }
        self.conv = nn.Conv1d(IC, OC, kernel, stride=stride).to(device=device, dtype=dtype, memory_format=memory_format)
        self.set_module_name("Conv1d")

    def forward(self, input):
        return self.conv(input)


class ConvTranspose1dBenchmark(ConvBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, L, device, dtype, memory_format):
        self.inputs = {"input": torch.rand(N, IC, L, device=device, dtype=dtype)}
        self.conv = nn.ConvTranspose1d(IC, OC, kernel, stride=stride).to(device=device, dtype=dtype, memory_format=memory_format)
        self.set_module_name("ConvTranspose1d")

    def forward(self, input):
        return self.conv(input)


op_bench.generate_pt_test(
    configs.conv_1d_configs_short + configs.conv_1d_configs_long, Conv1dBenchmark
)
op_bench.generate_pt_gradient_test(
    configs.conv_1d_configs_short + configs.conv_1d_configs_long, Conv1dBenchmark
)
op_bench.generate_pt_test(
    configs.conv_1d_configs_short + configs.conv_1d_configs_long,
    ConvTranspose1dBenchmark,
)


"""
Microbenchmarks for Conv2d, ConvTranspose2d, and Conv2dPointwise operators.
"""


class Conv2dBenchmark(ConvBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, H, W, G, pad, device, dtype, memory_format):
        self.inputs = {"input": torch.rand(N, IC, H, W, device=device, dtype=dtype)}
        self.conv = nn.Conv2d(IC, OC, kernel, stride=stride, groups=G, padding=pad).to(
            device=device,
            dtype=dtype,
            memory_format=memory_format,
        )
        self.set_module_name("Conv2d")

    def forward(self, input):
        return self.conv(input)


class ConvTranspose2dBenchmark(ConvBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, H, W, G, pad, device, dtype, memory_format):
        self.inputs = {"input": torch.rand(N, IC, H, W, device=device, dtype=dtype)}
        self.conv = nn.ConvTranspose2d(
            IC, OC, kernel, stride=stride, groups=G, padding=pad
        ).to(device=device, dtype=dtype, memory_format=memory_format)
        self.set_module_name("ConvTranspose2d")

    def forward(self, input):
        return self.conv(input)


class Conv2dPointwiseBenchmark(ConvBenchmarkBase):
    def init(self, IC, OC, stride, N, H, W, G, pad, device, dtype, memory_format):
        self.inputs = {"input": torch.rand(N, IC, H, W, device=device, dtype=dtype)}
        # Use 1 as kernel for pointwise convolution
        self.conv = nn.Conv2d(IC, OC, 1, stride=stride, groups=G, padding=pad).to(
            device=device,
            dtype=dtype,
            memory_format=memory_format,
        )
        self.set_module_name("Conv2dPointwise")

    def forward(self, input):
        return self.conv(input)


op_bench.generate_pt_test(
    configs.conv_2d_configs_short + configs.conv_2d_configs_long, Conv2dBenchmark
)
op_bench.generate_pt_gradient_test(
    configs.conv_2d_configs_short + configs.conv_2d_configs_long, Conv2dBenchmark
)
op_bench.generate_pt_test(
    configs.conv_2d_configs_short + configs.conv_2d_configs_long,
    ConvTranspose2dBenchmark,
)
op_bench.generate_pt_test(
    configs.conv_2d_pw_configs_short + configs.conv_2d_pw_configs_long,
    Conv2dPointwiseBenchmark,
)


"""
Microbenchmarks for Conv3d and ConvTranspose3d operators.
"""


class Conv3dBenchmark(ConvBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, D, H, W, device, dtype):
        self.inputs = {"input": torch.rand(N, IC, D, H, W, device=device, dtype=dtype)}
        self.conv = nn.Conv3d(IC, OC, kernel, stride=stride).to(device=device, dtype=dtype)
        self.set_module_name("Conv3d")

    def forward(self, input):
        return self.conv(input)


class ConvTranspose3dBenchmark(ConvBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, D, H, W, device, dtype):
        self.inputs = {"input": torch.rand(N, IC, D, H, W, device=device, dtype=dtype)}
        self.conv = nn.ConvTranspose3d(IC, OC, kernel, stride=stride).to(device=device, dtype=dtype)
        self.set_module_name("ConvTranspose3d")

    def forward(self, input):
        return self.conv(input)


op_bench.generate_pt_test(configs.conv_3d_configs_short, Conv3dBenchmark)
op_bench.generate_pt_gradient_test(configs.conv_3d_configs_short, Conv3dBenchmark)
# FIXME(lms): ConvTranposed3D is not supported on musa.
# op_bench.generate_pt_test(configs.conv_3d_configs_short, ConvTranspose3dBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
