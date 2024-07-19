import operator_benchmark as op_bench
import torch
import torch_musa

rmsnorm_long_configs = op_bench.cross_product_configs(
    input_shape=(
        (2, 128, 1024),
        (8, 32, 2048),
        (8, 128, 2048),
        (1, 128, 4096)
    ),
    device=["musa"],
    dtype=[torch.half],
    tags=["short"],
)


class RMSNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, input_shape, device, dtype):
        input_tenor = torch.randn(
            input_shape, dtype=dtype, device=device, requires_grad=self.auto_set()
        )
        # Now only test RMSNorm on the last dim
        weight_tensor = torch.randn(
            input_shape[-1], dtype=dtype, device=device, requires_grad=self.auto_set()
        )
        # inputs to operator
        self.inputs = {
            "input": input_tenor,
            "normalized_shape": [input_shape[-1]],
            "weight": weight_tensor,
            "eps": 1e-6,
        }

    def forward(self, input, normalized_shape, weight, eps):
        return torch.rms_norm(input, normalized_shape, weight, eps)


op_bench.generate_pt_test(rmsnorm_long_configs, RMSNormBenchmark)
op_bench.generate_pt_gradient_test(rmsnorm_long_configs, RMSNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
