from functools import reduce
import torch

import operator_benchmark as op_bench

"""Microbenchmarks for indexing operator, include index and index_put"""

indexing_configs = op_bench.cross_product_configs(
    config=[
        # input_shape, value_shape [ignored when benchmarking index], index_shape
        [(16, 10240), (4, 10240), ((4,),)],
        [(16, 16, 1024), (4, 1024), ((4,), (4,))],
    ],
    dtype=[torch.float32, torch.float16],
    device=["musa"],
    tags=["short"],
    op=["index", "index_put"],
)

class IndexingBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, config, dtype, device, op):
        indices = []
        for i, index_shape in enumerate(config[-1]):
            if index_shape:
                index_num = reduce((lambda x, y: x * y), index_shape)
                indices.append(
                    torch.randperm(config[0][i])[:index_num]
                    .to(torch.int64).to(device).reshape(index_shape)
                )
            else:
                indices.append(slice(None))
        self.inputs = {
            "inputs": torch.randn(config[0], dtype=dtype, device=device),
            "indices": indices
        }
        if op == "index_put":
            values = torch.randn(config[1], dtype=dtype, device=device)
            self.inputs.update({"values": values})

        self.set_module_name(f"{op}")

    def forward(self, inputs, indices, values=None):
        if values is not None:
            return self.run_index_put(inputs, indices, values)
        else:
            return self.run_index(inputs, indices)

    def run_index_put(self, inputs, indices, values):
        inputs[indices] = values
        return inputs

    def run_index(self, inputs, indices):
        out = inputs[indices]
        return out

op_bench.generate_pt_test(indexing_configs, IndexingBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
