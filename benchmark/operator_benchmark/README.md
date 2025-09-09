# Operator Benchmark

This is a benchmark suite for TorchMusa copied from PyTorch. It is almost the same as [what torch have](https://github.com/pytorch/pytorch/tree/main/benchmarks/operator_benchmark), except that we modify it to support `musa` as the backend device. So you can click the [link](https://github.com/pytorch/pytorch/tree/main/benchmarks/operator_benchmark) to get more details about `operator_benchmark`.

Because `operator_benchmark` is not a part of released python package of PyTorch, we have made a copy to torch_musa repo to use it.

## Features added to PyTorch operator benchmark

1. Supported device: cpu, musa


## Getting Started

We should install a cpp_extension for PyTorch first to run torch_musa operator benchmark:
```
cd $PYTORCH_REPO/benchmarks/operator_benchmark/pt_extension
python setup.py install
```

You can run benchmarks for different operators now in `benchmark/operator_benchmark/tests`. For example, we have already added a benchmark test for `RMSNorm` in `rmsnorm_test.py`:

```
cd $TORCH_MUSA_REPO/benchmark/operator_benchmark
python -m tests.rmsnorm_test
``

Output will be like below:

```plaintext
# ----------------------------------------
# PyTorch/Caffe2 Operator Micro-benchmarks
# ----------------------------------------
# Tag : short

# Benchmarking PyTorch: RMSNormBenchmark
# Mode: Eager
# Name: RMSNormBenchmark_input_shape(2,128,1024)_musa_dtypetorch.float16
# Input: input_shape: (2, 128, 1024), device: musa, dtype: torch.float16
Forward Execution Time (us) : 29.381

# Benchmarking PyTorch: RMSNormBenchmark
# Mode: Eager
# Name: RMSNormBenchmark_input_shape(2,128,1024)_musa_dtypetorch.float32
# Input: input_shape: (2, 128, 1024), device: musa, dtype: torch.float32
Forward Execution Time (us) : 30.456

# Benchmarking PyTorch: RMSNormBenchmark
# Mode: Eager
# Name: RMSNormBenchmark_input_shape(8,32,2048)_musa_dtypetorch.float16
# Input: input_shape: (8, 32, 2048), device: musa, dtype: torch.float16
Forward Execution Time (us) : 34.333

# Benchmarking PyTorch: RMSNormBenchmark
# Mode: Eager
# Name: RMSNormBenchmark_input_shape(8,32,2048)_musa_dtypetorch.float32
# Input: input_shape: (8, 32, 2048), device: musa, dtype: torch.float32
Forward Execution Time (us) : 35.437
```

Outputs will contain the execution time of operator with specified inputs and attributes.

To add an operator benchmark test, you could borrow the test which is already in PyTorch repo. For operators which exist only in `musa` backend, check `rmsnorm_test.py` as a reference to add benchmark tests for them.


## Get the benchmark result

### Get the benchmark result at runtime

Actually you could get the benchmark result at runtime using `BenchmarkRunner` API. For example:

```python
# create a benchmark runner 
runner = BenchmarkRunner(*args)
benchmark_result = runner.run()
```

`benchmark_result` will be a dict with formatted benchmark result of each test case.

### Save the benchmark result to disk

Benchmark results could be saved to disk as a JSON file. To save the output of operator benchmarks, you need to pass the argument `--res-dir` (the directory to store the results) and `--res-file-name` (the filename of the saved results). Below is sample output of operator benchmark of `RMSNorm`:

```bash
python -m tests.rmsnorm_test --res-dir DIR_SPECIFIED --res-file-name benchmark_test
```
Result file `${DIR_SPECIFIED}/benchmark_test.json`:
```json
{
  // MUSA软件栈信息
  "musa_stack": {
    "musa": {
      "musa_runtime": {
        "musa_runtime_info": {
          "version": "2.1.0",
          "gitbranch": "rc2.1.0",
          "gittag": "Notag",
          "commitid": "871ff3c18bb06e3c521275b0e7732b674cddb6dd",
          "commitdate": "2024-03-2515:29:03+0800"
        },
        "driver_depends_info": {
          "gitbranch": "heads/20240320_develop",
          "gittag": "20240320_develop",
          "commitid": "4f591d074070595db19003e5069e78a7ff8942d1",
          "commitdate": "2024-03-2015:28:45+0800"
        }
      },
      "musa_toolkits": {
        "version": "2.1.0",
        "gitbranch": "rc2.1.0",
        "gittag": "Notag",
        "commitid": "a3ce975dc82796d7ce21fc1227d36f670eba895d",
        "commitdate": "2024-04-0914:38:18+0800"
      },
      "mudnn": {
        "version": "2.5.0",
        "gitbranch": "heads/release2.5.0",
        "gittag": "20240325_develop",
        "commitid": "fc5c421516ee3d013bc3a9d23ffdc6bc06f9ee79",
        "commitdate": "2024-03-2214:37:29+0800"
      }
    },
    "driver": {
      "ddk_date": "20240325",
      "ddk_type": "release",
      "ddk_version": "Release_24.04",
      "ddk_commit": "4f591d074@20240320"
    }
  },

  "benchmark_result": [
    {
      "op": "RMSNormBenchmark", // benchmark type
      "mode": "Eager", // JIT or Eager
      "test_cases": [ // list of different cases of RMSNorm benchmark
        {
          "test_name": "RMSNormBenchmark_input_shape(2,128,1024)_musa_dtypetorch.float16", // benchmark case名
          "latency": 29.563, // 算子运行耗时的中位数
          "unit": "us", // 耗时单位
          "macs": -1, // 算子理论运算指令数
          "tflops": 0, // FLOPS
          "backward": false, // 是否为梯度算子
          "time_metric": [
            29.7, // N次算子运行耗时的均值
            0.088, //  N次算子运行耗时的方差
            [
              29.489,// N次算子运行耗时的 %0 分位数
              29.527,// N次算子运行耗时的 %25 分位数
              29.563,// N次算子运行耗时的 %50 分位数
              29.732,// N次算子运行耗时的 %75 分位数
              30.509,// N次算子运行耗时的 %100 分位数
            ]
          ],
          "test_config": { // 本次benchamrk case的config
            "input_shape": [
              2,
              128,
              1024
            ],
            "device": "musa",
            "dtype": "torch.float16"
          }
        }
      ]
    },
    {
      "op": "RMSNormBenchmark",
      "mode": "Eager",
      "test_cases": [
        {
          "test_name": "RMSNormBenchmark_input_shape(2,128,1024)_musa_dtypetorch.float16_bwdall",
          "latency": 220.068,
          "unit": "us",
          "macs": -1,
          "tflops": 0,
          "backward": true,
          "time_metric": [
            225.691,
            183.877,
            [
              202.727,
              219.432,
              220.068,
              236.458,
              245.264
            ]
          ],
          "test_config": {
            "input_shape": [
              2,
              128,
              1024
            ],
            "device": "musa",
            "dtype": "torch.float16"
          }
        },
        {
          "test_name": "RMSNormBenchmark_input_shape(2,128,1024)_musa_dtypetorch.float16_bwd1",
          "latency": 199.307,
          "unit": "us",
          "macs": -1,
          "tflops": 0,
          "backward": true,
          "time_metric": [
            198.805,
            1.869,
            [
              195.633,
              198.733,
              199.307,
              199.607,
              200.018
            ]
          ],
          "test_config": {
            "input_shape": [
              2,
              128,
              1024
            ],
            "device": "musa",
            "dtype": "torch.float16"
          }
        },
        {
          "test_name": "RMSNormBenchmark_input_shape(2,128,1024)_musa_dtypetorch.float16_bwd2",
          "latency": 199.469,
          "unit": "us",
          "macs": -1,
          "tflops": 0,
          "backward": true,
          "time_metric": [
            199.879,
            4.816,
            [
              197.19,
              198.757,
              199.469,
              200.13,
              204.718
            ]
          ],
          "test_config": {
            "input_shape": [
              2,
              128,
              1024
            ],
            "device": "musa",
            "dtype": "torch.float16"
          }
        }
      ]
    }
  ]
}
```

## Visualization

The JSON output file could be parsed for convenient analysis, but not friendly for visualization.

A simple [visiulization tool](./utils/op_bench_json2csv.py) for the JSON file is provided. This tool simply converts a JSON result file to a CSV, in which each line represents a test case. 

**Mark:** This visualization tool just converts the configurations of each test case to a dict string in CSV.


Usage:

```bash
python op_bench_json2csv.py ${OUTPUT_JSON_FILE_OF_OP_BENCHMARK}
```
Example result of a CSV output:

```csv
OP               ,Mode  ,TestCase                                                                  ,TestConfig                                                                    ,backward ,macs ,tflops ,Latency ,latency_variance ,latency_mean ,0%      ,25%     ,50%     ,75%     ,100%
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(2,128,1024)_musa_dtypetorch.float16"        ,"{'input_shape': [2, 128, 1024], 'device': 'musa', 'dtype': 'torch.float16'}" ,False    ,-1   ,     0 , 29.118 ,           0.069 ,      29.219 , 29.042 , 29.072 , 29.118 , 29.166 , 29.946
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(8,32,2048)_musa_dtypetorch.float16"         ,"{'input_shape': [8, 32, 2048], 'device': 'musa', 'dtype': 'torch.float16'}"  ,False    ,-1   ,     0 , 33.784 ,           0.065 ,      33.912 , 33.736 , 33.758 , 33.784 , 33.924 , 34.588
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(8,128,2048)_musa_dtypetorch.float16"        ,"{'input_shape': [8, 128, 2048], 'device': 'musa', 'dtype': 'torch.float16'}" ,False    ,-1   ,     0 , 47.877 ,           0.041 ,      47.962 , 47.829 , 47.84  , 47.877 , 48.018 , 48.493
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(1,128,4096)_musa_dtypetorch.float16"        ,"{'input_shape': [1, 128, 4096], 'device': 'musa', 'dtype': 'torch.float16'}" ,False    ,-1   ,     0 , 41.593 ,           0.047 ,      41.695 , 41.541 , 41.545 , 41.593 , 41.692 , 42.239
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(2,128,1024)_musa_dtypetorch.float16_bwdall" ,"{'input_shape': [2, 128, 1024], 'device': 'musa', 'dtype': 'torch.float16'}" ,True     ,-1   ,     0 ,195.865 ,         322.567 ,     204.621 ,193.62  ,193.973 ,195.865 ,203.875 ,247.161
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(2,128,1024)_musa_dtypetorch.float16_bwd1"   ,"{'input_shape': [2, 128, 1024], 'device': 'musa', 'dtype': 'torch.float16'}" ,True     ,-1   ,     0 ,194.261 ,           0.13  ,     194.154 ,193.415 ,194.023 ,194.261 ,194.366 ,194.627
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(2,128,1024)_musa_dtypetorch.float16_bwd2"   ,"{'input_shape': [2, 128, 1024], 'device': 'musa', 'dtype': 'torch.float16'}" ,True     ,-1   ,     0 ,193.996 ,           0.188 ,     193.874 ,193.044 ,193.677 ,193.996 ,194.139 ,194.446
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(8,32,2048)_musa_dtypetorch.float16_bwdall"  ,"{'input_shape': [8, 32, 2048], 'device': 'musa', 'dtype': 'torch.float16'}"  ,True     ,-1   ,     0 ,214.838 ,           0.26  ,     214.932 ,214.025 ,214.703 ,214.838 ,215.33  ,215.59
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(8,32,2048)_musa_dtypetorch.float16_bwd1"    ,"{'input_shape': [8, 32, 2048], 'device': 'musa', 'dtype': 'torch.float16'}"  ,True     ,-1   ,     0 ,212.984 ,           0.376 ,     212.865 ,211.478 ,212.864 ,212.984 ,213.168 ,213.527
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(8,32,2048)_musa_dtypetorch.float16_bwd2"    ,"{'input_shape': [8, 32, 2048], 'device': 'musa', 'dtype': 'torch.float16'}"  ,True     ,-1   ,     0 ,212.607 ,           0.238 ,     212.679 ,211.908 ,212.349 ,212.607 ,213.086 ,213.368
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(8,128,2048)_musa_dtypetorch.float16_bwdall" ,"{'input_shape': [8, 128, 2048], 'device': 'musa', 'dtype': 'torch.float16'}" ,True     ,-1   ,     0 ,296.904 ,           0.083 ,     296.936 ,296.412 ,296.803 ,296.904 ,297.133 ,297.361
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(8,128,2048)_musa_dtypetorch.float16_bwd1"   ,"{'input_shape': [8, 128, 2048], 'device': 'musa', 'dtype': 'torch.float16'}" ,True     ,-1   ,     0 ,295.747 ,           0.275 ,     295.591 ,294.457 ,295.464 ,295.747 ,295.906 ,296.193
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(8,128,2048)_musa_dtypetorch.float16_bwd2"   ,"{'input_shape': [8, 128, 2048], 'device': 'musa', 'dtype': 'torch.float16'}" ,True     ,-1   ,     0 ,295.371 ,           0.024 ,     295.416 ,295.145 ,295.361 ,295.371 ,295.494 ,295.687
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(1,128,4096)_musa_dtypetorch.float16_bwdall" ,"{'input_shape': [1, 128, 4096], 'device': 'musa', 'dtype': 'torch.float16'}" ,True     ,-1   ,     0 ,221.199 ,           0.127 ,     221.237 ,220.813 ,220.886 ,221.199 ,221.591 ,221.692
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(1,128,4096)_musa_dtypetorch.float16_bwd1"   ,"{'input_shape': [1, 128, 4096], 'device': 'musa', 'dtype': 'torch.float16'}" ,True     ,-1   ,     0 ,221.18  ,           0.107 ,     221.038 ,220.281 ,221.032 ,221.18  ,221.203 ,221.334
RMSNormBenchmark ,Eager ,"RMSNormBenchmark_input_shape(1,128,4096)_musa_dtypetorch.float16_bwd2"   ,"{'input_shape': [1, 128, 4096], 'device': 'musa', 'dtype': 'torch.float16'}" ,True     ,-1   ,     0 ,220.876 ,           0.969 ,     220.449 ,218.155 ,220.437 ,220.876 ,221.034 ,221.167
```