# unit test for PyTorch
reference: test_binary.py

Test examples:
```bash
pytest -s ./
pytest -s test_binary.py
pytest -s test_bainary.py::test_add
```

## Note
- filename must begin with "test_"
- function name must begin with "test_"
- please add tests as much as possible, including scalar/zero-elements/multi-dimension/multiple data types.

## How to add new unit test for operator
using `testing.OpTest` to test operator is recommended  
### test function or method
We test `torch.xxx` and `torch.nn.functional.xxx` in most cases, below are some examples to clarify how to use `testing.Optest` to test these operators.  
#### create dtype of tensor to be tested expicitly [Highly Recommended]
In this case, OpTest doesn't know which dtype to test, we should pass correct tensor to OpTest.
```python
func = torch.abs
dtype = torch.bfloat16
input_data = {"input": torch.randn((2,3,4), dtype=dtype)}
test = testing.OpTest(func=func, input_args=input_data)
test.check_result()
```

#### specify `test_dtype` only
In this case, it doesn't matter what the dtype of tensor is here, OpTest will takes care of everything by specifying `test_dtype`.


**NOTE: If you are using `test_dtype`, make sure the operator's inputs are all of the same dtype, otherwise consider using `dtype_nocast_map` to disable the dtype conversion of some tensors. See the next section for an example**
```python
func = torch.abs
test_dtype = torch.bfloat16

# it doesn't matter what the dtype of tensor is here, OpTest will takes care of everything
input_data = {"input": torch.randn((2,3,4))}
test = testing.OpTest(func=func, input_args=input_data, test_dtype=test_dtype)
test.check_result()
```

#### specify `test_dtype` and `dtype_nocast_map`
In this case, the indices' dtype have nothing to do with `float_dtypes`, thus if we have specified `test_dtype`, we should also disable the dtype conversion of `indices` by specifying `dtype_nocast_map`.
```python
@pytest.mark.parametrize("input_shape", input_shapes)
@pytest.mark.parametrize("dtype", float_dtypes)
def test_embedding_bwd(input_shape, dtype):
    global m, n
    grad_output = torch.randn((*input_shape, m))
    indices = torch.randint(low=0, high=n, size=input_shape)
    input_args = {
        "grad_output": grad_output,
        "indices": indices,
        "num_weights": n,
        "padding_idx": -1,
        "scale_grad_by_freq": False,
    }
    test = testing.OpTest(
        func=torch.ops.aten.embedding_dense_backward,
        input_args=input_args,
        comparators=comparator,
        test_dtype=dtype,
    )
    test.check_result(dtype_nocast_map={"indices": True})
```

### test nn.Module
When testing `nn.Module` which has learnable parameters such as `nn.Conv2d`, we should be careful that `test_dtype` may need to be specified or we should initialize `nn.Module` explicitly.

**NOTE: It is better for users to add unit tests like CASE2.**
```python
#################### CASE1 ####################
test_dtype = torch.bfloat16
func = torch.nn.Conv2d
input_args = {"in_channels": 3, "out_channels": 16, "kernel_size": 3}
test = testing.OpTest(func=func, input_args=input_args, test_dtype=test_dtype)
input_data = {"input": torch.randn(2, 3, 16, 16)}  # it doesn't matter what the dtype of tensor is here 
test.check_result(input_data)

#################### CASE2 ####################
test_dtype = torch.bfloat16
conv_args = {"in_channels": 3, "out_channels": 16, "kernel_size": 3}
func = torch.nn.Conv2d(**conv_args).to(test_dtype)
test = testing.OpTest(func=func, input_args={"input": torch.randn(2, 3, 16, 16).to(test_dtype)})
test.check_result()
```
Of course, you can call `torch.ops.aten.convolution` instead of `nn.Conv2d`, then the test method degrades to the previous section.
