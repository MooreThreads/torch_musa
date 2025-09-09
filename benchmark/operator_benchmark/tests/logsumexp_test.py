import operator_benchmark as op_bench
import torch
import torch_musa

logsumexp_short_configs = op_bench.cross_product_configs(
    input_shape=(
        (2, 128, 512),
        (1, 128, 512),
        (8, 128, 512),
        (16, 256, 512),
    ),
    dim=[1, 2],
    device=["musa"],
    dtype=[torch.float32, torch.half],
    tags=["short"],
)


class LogsumexpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, input_shape, device, dim, dtype):
        input_tenor = torch.randn(
            input_shape, dtype=dtype, device=device, requires_grad=self.auto_set()
        )
        # inputs to operator
        self.inputs = {
            "input": input_tenor,
            "dim": dim,
        }

    def forward(self, input, dim):
        return torch.logsumexp(input, dim=dim)


op_bench.generate_pt_test(logsumexp_short_configs, LogsumexpBenchmark)
op_bench.generate_pt_gradient_test(logsumexp_short_configs, LogsumexpBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
