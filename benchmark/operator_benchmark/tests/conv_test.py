import operator_benchmark as op_bench
import torch
import torch.nn as nn
import numpy as np

from tests import configs

"""
Microbenchmarks for Conv1d and ConvTranspose1d operators.
"""


class ConvBenchmarkBase(op_bench.TorchBenchmarkBase):
    def __init__(self):
        super().__init__()
        self.conv = None

    def calc_flops(self):
        super().calc_flops()
        x_input = self.inputs["input"]
        batch_size = x_input.shape[0]
        assert self.outputs is not None
        assert self.conv is not None
        output = self.outputs[0]
        output_dims = list(output.shape[2:])

        kernel_dims = list(self.conv.kernel_size)
        in_channels = self.conv.in_channels
        out_channels = self.conv.out_channels
        groups = self.conv.groups

        filters_per_channel = out_channels // groups
        conv_per_position_flops = int(np.prod(kernel_dims, dtype=np.int64)) * (
            in_channels * filters_per_channel
        )

        active_elements_count = batch_size * int(np.prod(output_dims, dtype=np.int64))

        overall_conv_flops = conv_per_position_flops * active_elements_count

        bias_flops = 0

        if self.conv.bias is not None:

            bias_flops = out_channels * active_elements_count

        overall_flops = overall_conv_flops + bias_flops

        return int(overall_flops)


class Conv1dBenchmark(ConvBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, L, device):
        self.inputs = {
            "input": torch.rand(N, IC, L, device=device, requires_grad=self.auto_set())
        }
        self.conv = nn.Conv1d(IC, OC, kernel, stride=stride).to(device=device)
        self.set_module_name("Conv1d")

    def forward(self, input):
        return self.conv(input)


class ConvTranspose1dBenchmark(ConvBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, L, device):
        self.inputs = {"input": torch.rand(N, IC, L, device=device)}
        self.conv = nn.ConvTranspose1d(IC, OC, kernel, stride=stride).to(device=device)
        self.set_module_name("ConvTranspose1d")

    def forward(self, input):
        return self.conv(input)


op_bench.generate_pt_test(
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
    def init(self, IC, OC, kernel, stride, N, H, W, G, pad, device):
        self.inputs = {"input": torch.rand(N, IC, H, W, device=device)}
        self.conv = nn.Conv2d(IC, OC, kernel, stride=stride, groups=G, padding=pad).to(
            device=device
        )
        self.set_module_name("Conv2d")

    def forward(self, input):
        return self.conv(input)


class ConvTranspose2dBenchmark(ConvBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, H, W, G, pad, device):
        self.inputs = {"input": torch.rand(N, IC, H, W, device=device)}
        self.conv = nn.ConvTranspose2d(
            IC, OC, kernel, stride=stride, groups=G, padding=pad
        ).to(device=device)
        self.set_module_name("ConvTranspose2d")

    def forward(self, input):
        return self.conv(input)


class Conv2dPointwiseBenchmark(ConvBenchmarkBase):
    def init(self, IC, OC, stride, N, H, W, G, pad, device):
        self.inputs = {"input": torch.rand(N, IC, H, W, device=device)}
        # Use 1 as kernel for pointwise convolution
        self.conv = nn.Conv2d(IC, OC, 1, stride=stride, groups=G, padding=pad).to(
            device=device
        )
        self.set_module_name("Conv2dPointwise")

    def forward(self, input):
        return self.conv(input)


op_bench.generate_pt_test(
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
    def init(self, IC, OC, kernel, stride, N, D, H, W, device):
        self.inputs = {"input": torch.rand(N, IC, D, H, W, device=device)}
        self.conv = nn.Conv3d(IC, OC, kernel, stride=stride).to(device=device)
        self.set_module_name("Conv3d")

    def forward(self, input):
        return self.conv(input)


class ConvTranspose3dBenchmark(ConvBenchmarkBase):
    def init(self, IC, OC, kernel, stride, N, D, H, W, device):
        self.inputs = {"input": torch.rand(N, IC, D, H, W, device=device)}
        self.conv = nn.ConvTranspose3d(IC, OC, kernel, stride=stride).to(device=device)
        self.set_module_name("ConvTranspose3d")

    def forward(self, input):
        return self.conv(input)


op_bench.generate_pt_test(configs.conv_3d_configs_short, Conv3dBenchmark)
# FIXME(lms): ConvTranposed3D is not supported on musa.
# op_bench.generate_pt_test(configs.conv_3d_configs_short, ConvTranspose3dBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
