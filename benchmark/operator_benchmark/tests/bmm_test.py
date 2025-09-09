import torch
import numpy as np

import operator_benchmark as op_bench

"""Microbenchmarks for add_ operator. Supports both Caffe2/PyTorch."""


class BmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, K, device, op):
        self.inputs = {
            "batch1": torch.rand(
                (B, M, K), device=device, requires_grad=self.auto_set()
            ),
            "batch2": torch.rand(
                (
                    B,
                    K,
                    N,
                ),
                device=device,
                requires_grad=self.auto_set(),
            ),
        }
        self.set_module_name(f"bmm (actual op={op}")
        self.op = torch.bmm if op == "bmm" else torch.matmul

    def calc_flops(self):
        super().calc_flops()
        batch_1 = self.inputs["batch1"]
        batch_2 = self.inputs["batch2"]
        assert batch_1.dim() == batch_2.dim() == 3
        # bmm ([B, M, K] @ [B, K, N] -> [B, M, N]), FLOPs: 2 * B * M * K * N
        flops = 2 * np.prod(batch_1.shape, dtype=np.int64) * batch_2.shape[-1]

        return int(flops)

    def calc_memory(self):
        super().calc_memory()
        batch_1 = self.inputs["batch1"]
        batch_2 = self.inputs["batch2"]
        output = self.outputs[0]
        assert batch_1.dim() == batch_2.dim() == 3
        if batch_1.dtype in [torch.float16, torch.bfloat16]:
            byte_val = 2
        else:
            byte_val = 4

        macs = byte_val * (
            np.prod(batch_1.shape) + np.prod(batch_2.shape) + np.prod(output.shape)
        )
        return int(macs)

    def forward(self, batch1, batch2):
        return self.op(batch1, batch2)


bmm_configs = op_bench.cross_product_configs(
    B=[16, 128],
    M=[64, 256],
    N=[256, 64],
    K=[64, 32],
    device=["musa"],
    tags=["short"],
    op=["bmm", "matmul"],
)

op_bench.generate_pt_test(bmm_configs, BmmBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
