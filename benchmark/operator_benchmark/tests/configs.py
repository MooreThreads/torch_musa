import operator_benchmark as op_bench
import torch

"""
Configs shared by multiple benchmarks
"""


def remove_musa(config_list):
    musa_config = {"device": "musa"}
    return [config for config in config_list if musa_config not in config]


# Configs for conv-1d ops
conv_1d_configs_short = op_bench.config_list(
    attr_names=["IC", "OC", "kernel", "stride", "N", "L"],
    attrs=[
        [256, 256, 3, 2, 4, 64],
    ],
    cross_product_configs={
        "device": ["musa"],
        "dtype": [torch.float32, torch.float16],
        "memory_format": [torch.contiguous_format, torch.channels_last],
    },
    tags=["short"],
)

conv_1d_configs_long = op_bench.cross_product_configs(
    IC=[128, 512],
    OC=[128, 512],
    kernel=[3],
    stride=[1, 2],
    N=[8],
    L=[128],
    device=["musa"],
    dtype=[torch.float32, torch.float16],
    memory_format=[torch.contiguous_format, torch.channels_last],
    tags=["long"],
)

# Configs for Conv2d and ConvTranspose2d
conv_2d_configs_short = op_bench.config_list(
    attr_names=[
        "IC",
        "OC",
        "kernel",
        "stride",
        "N",
        "H",
        "W",
        "G",
        "pad",
    ],
    attrs=[
        [512, 256, 3, 1, 8, 32, 32, 1, 0],
        [512, 1024, 3, 1, 64, 16, 16, 1, 0],
        [1024, 1024, 3, 1, 128, 16, 16, 1, 0],
        [1024, 2048, 3, 1, 128, 32, 32, 1, 0],
    ],
    cross_product_configs={
        "device": ["musa"],
        "dtype": [torch.float32, torch.float16],
        "memory_format": [torch.contiguous_format, torch.channels_last],
    },
    tags=["short"],
)

conv_2d_configs_long = op_bench.cross_product_configs(
    IC=[128, 256, 512],
    OC=[256, 512, 1024],
    kernel=[3],
    stride=[1, 2],
    N=[4, 16, 64],
    H=[32, 64],
    W=[32, 64],
    G=[1],
    pad=[0],
    device=["musa"],
    dtype=[torch.float32, torch.float16],
    memory_format=[torch.contiguous_format, torch.channels_last],
    tags=["long"],
)

# Configs for Conv2dPointwise
conv_2d_pw_configs_short = op_bench.config_list(
    attr_names=[
        "IC",
        "OC",
        "stride",
        "N",
        "H",
        "W",
        "G",
        "pad",
    ],
    attrs=[
        [512, 512, 1, 8, 16, 16, 1, 0],
        [1024, 512, 1, 128, 16, 16, 1, 0],
        [2048, 1024, 1, 256, 32, 32, 1, 0],
    ],
    cross_product_configs={
        "device": ["musa"],
        "dtype": [torch.float32, torch.float16],
        "memory_format": [torch.contiguous_format, torch.channels_last],
    },
    tags=["short"],
)

conv_2d_pw_configs_long = op_bench.cross_product_configs(
    IC=[128, 256, 512],
    OC=[256, 512, 1024],
    stride=[1, 2],
    N=[4, 16, 64],
    H=[32, 64],
    W=[32, 64],
    G=[1],
    pad=[0],
    device=["musa"],
    dtype=[torch.float32, torch.float16],
    memory_format=[torch.contiguous_format, torch.channels_last],
    tags=["long"],
)

# Configs for Conv3d and ConvTranspose3d
conv_3d_configs_short = op_bench.config_list(
    attr_names=["IC", "OC", "kernel", "stride", "N", "D", "H", "W"],
    attrs=[
        [64, 64, 3, 1, 8, 4, 16, 16],
        [128, 256, 3, 1, 16, 4, 32, 32],
    ],
    cross_product_configs={
        "device": ["musa"],
        "dtype": [torch.float32, torch.float16]
    },
    tags=["short"],
)

linear_configs_short = op_bench.config_list(
    attr_names=["IN", "OC"],
    attrs=[
        [[8, 1024, 2048], 2048],
        [[8, 4096, 4096], 4096],
        [[8, 4096, 4096], 11008],
        [[8, 3584, 3854], 18944],
    ],
    cross_product_configs={
        "BIAS": [True, False],
        "device": ["musa"],
        "dtype": [torch.float32, torch.float16],
    },
    tags=["short"],
)


linear_configs_long = op_bench.cross_product_configs(
    IN=[[16, 512], [128, 1024]],
    OC=[64, 128],
    BIAS=[True, False],
    device=["cpu", "musa"],
    dtype=[torch.float32, torch.float16],
    tags=["long"],
)

embeddingbag_short_configs = op_bench.cross_product_configs(
    embeddingbags=[10, 120, 1000, 2300],
    dim=[64],
    mode=["sum"],
    input_size=[8, 16, 64],
    offset=[0],
    sparse=[True, False],
    include_last_offset=[True, False],
    device=["cpu"],
    tags=["short"],
)

embedding_short_configs = op_bench.cross_product_configs(
    num_embeddings=[10, 120, 1000, 2300],
    embedding_dim=[64],
    input_size=[8, 16, 64],
    device=["cpu"],
    tags=["short"],
)
