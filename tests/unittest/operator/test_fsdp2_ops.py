# pylint: disable=all
import pytest
import copy
import math
from functools import reduce

import torch
from torch_musa import testing


# used by torch.ops.fsdp.split_with_sizes_copy
# fmt: off
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("configs", [
    # output_dims, split_dim

    # llama3-8b case
    [[[4, 4194304], [4, 1048576], [4, 1048576], [4, 4194304], [4, 14680064], [4, 14680064], [4, 14680064], [4, 1024], [4, 1024]], 1],
    [[[4, 128], [4, 256], [4, 512]], 1],
    [[[4, 128, 6], [4, 256, 6], [4, 512, 6]], 1],
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_split_with_sizes_copy_out(configs, dtype):
    output_dims = configs[0]
    split_dim = configs[1]

    # calculate dim of input tensor
    input_dims = copy.deepcopy(output_dims[0])
    split_sizes = [output_dims[0][split_dim],]
    for dims in output_dims[1:]:
        input_dims[split_dim] += dims[split_dim]
        split_sizes.append(dims[split_dim])
    gpu_device = torch.musa.current_device()
    input_tensor = torch.randn(input_dims).to(dtype)
    output_tensor_lst = [torch.randn(dims).to(dtype) for dims in output_dims]
    input_tensor_gpu = input_tensor.to(gpu_device)
    output_tensor_lst_gpu = [t.to(gpu_device) for t in output_tensor_lst]

    torch.split_with_sizes_copy(input_tensor_gpu, split_sizes, dim=split_dim, out=output_tensor_lst_gpu)
    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=split_dim, out=output_tensor_lst)

    test = testing.OpTest(func=torch.split_with_sizes_copy)
    test.compare_res(res1=output_tensor_lst, res2=output_tensor_lst_gpu)


# used by foreach_reduce_scatter_copy_in
# fmt: off
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("configs", [
    # input_dims, dim, num_chunks

    # llama3-8b case
    [[[4096, 4096], [1024, 4096], [1024, 4096], [4096, 4096], [14336, 4096], [4096, 14336], [14336, 4096], [4096], [4096]], 0, 4],

    # with padding
    [[[4, 4096], [3, 4096], [3, 4096]], 0, 4],
    [[[4, 8, 128], [4, 7, 256], [4, 4, 256]], 1, 2],
    [[[4, 8, 128], [4, 8, 256], [4, 4, 256]], 1, 2],

])
@pytest.mark.parametrize("dtype", [(torch.float16, torch.float32), (torch.bfloat16, torch.float32), (torch.bfloat16,), (torch.float32,)])
@pytest.mark.parametrize("test_out", [True, False])
def test_chunk_cat(configs, dtype, test_out):
    input_dims = configs[0]
    chunk_dim, num_chunks = configs[1], configs[2]
    if len(dtype) == 1:
        input_dtype = output_dtype = dtype[0]
    else:
        input_dtype, output_dtype = dtype
    gpu_device = torch.musa.current_device()

    input_tensors = [torch.randn(dims).to(input_dtype) for dims in input_dims]
    input_tensors_gpu = [t.to(gpu_device) for t in input_tensors]

    if test_out:
        trailing_numel = 0
        out_dims = input_dims[0][:chunk_dim]
        for dims in input_dims:
            chunk_size_along_dim = (dims[chunk_dim] + num_chunks - 1) // num_chunks
            trailing_numel += math.prod(dims[chunk_dim + 1 :]) * chunk_size_along_dim
        out_dims.extend([num_chunks, trailing_numel])
        output_tensor = torch.randn(out_dims).to(output_dtype)
        output_tensor_gpu = output_tensor.to(gpu_device)
    else:
        output_tensor_gpu = output_tensor = None

    # fmt: off
    out_gpu = torch._chunk_cat(input_tensors_gpu, chunk_dim, num_chunks, out=output_tensor_gpu)
    out_cpu = torch._chunk_cat(input_tensors, chunk_dim, num_chunks, out=output_tensor)

    test = testing.OpTest(func=torch._chunk_cat)
    test.compare_res(res1=[out_gpu.cpu(),], res2=[out_cpu,])

    if test_out:
        assert out_gpu.data_ptr() == output_tensor_gpu.data_ptr()
