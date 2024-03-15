"compare tool with cpu"
# pylint: disable=broad-exception-caught,broad-exception-raised
import pickle
import torch
from torch.utils._python_dispatch import TorchDispatchMode

def recursive_apply(func):
    'recursive apply func'
    def recursive_apply_fn(inputs):
        if isinstance(inputs, (list, tuple)):
            inputs_dst = [None] * len(inputs)
            for i, x in enumerate(inputs):
                inputs_dst[i] = recursive_apply_fn(x)
            if isinstance(inputs, tuple):
                inputs_dst = tuple(inputs_dst)
            return inputs_dst
        if isinstance(inputs, dict):
            inputs_dst = {}
            for k, v in inputs.items():
                inputs_dst[k] = recursive_apply_fn(v)
            return inputs_dst
        if isinstance(inputs, torch.Tensor):
            inputs_dst = func(inputs.clone().detach())
            return inputs_dst

        return inputs

    return recursive_apply_fn


def convert_to_dtype(inputs, dtype):
    return recursive_apply(lambda x: x.to(dtype))(inputs)


def convert_to_cpu(inputs):
    return recursive_apply(lambda x: x.cpu())(inputs)


def convert_to_musa(inputs):
    return recursive_apply(lambda x: x.to("musa"))(inputs)


def compare_tensors(tensor1, tensor2, atol, rtol):
    'compare two tensors and return a position mask where values are not closed'
    # Check NaN and Inf
    nan_mask1, nan_mask2 = torch.isnan(tensor1), torch.isnan(tensor2)
    inf_mask1, inf_mask2 = torch.isinf(tensor1), torch.isinf(tensor2)

    nan_diff = nan_mask1 != nan_mask2
    inf_diff = inf_mask1 != inf_mask2

    basic_diff = torch.abs(tensor1 - tensor2)
    tolerance = atol + rtol * torch.abs(tensor2)

    # Compare normal values
    normal_mask1, normal_mask2 = ~nan_mask1 & ~inf_mask1, ~nan_mask2 & ~inf_mask2
    normal_not_close = (basic_diff > tolerance) & normal_mask1 & normal_mask2

    # Aggregate the comparison results
    not_close = nan_diff | inf_diff | normal_not_close

    return not_close


def print_tensors_diff(tensor1, tensor2, atol, rtol):
    'print different values of two tensors'
    not_close = compare_tensors(
        tensor1.to(tensor2.device).to(tensor2.dtype), tensor2, atol, rtol
    )
    indices = torch.nonzero(not_close)
    indices_np = indices.cpu().numpy()

    # If the indices are too large, only process the front part
    if len(indices_np) > 20:
        print(f"\nToo many indices (total {len(indices_np)}) to print \n...")
        indices_np = indices_np[:20]

    idx_tuples = [tuple(idx) for idx in indices_np]
    elements_out1 = [tensor1[idx].item() for idx in idx_tuples]
    elements_out2 = [tensor2[idx].item() for idx in idx_tuples]

    for idx_tuple, elem1, elem2 in zip(idx_tuples, elements_out1, elements_out2):
        print(f"Element at index {idx_tuple} is not close: {elem1} vs {elem2}")

    print(
        f"\n\ntensor 1: shape={tensor1.shape},\
             numbers of nan = {torch.isnan(tensor1).sum().item()} of {tensor1.numel()},\
             numbers of inf = {torch.isinf(tensor1).sum().item()} of {tensor1.numel()}"
    )
    print(tensor1)
    print(
        f"\n\ntensor 2 (golden): shape={tensor2.shape},\
             numbers of nan = {torch.isnan(tensor2).sum().item()} of {tensor2.numel()},\
             numbers of inf = {torch.isinf(tensor2).sum().item()} of {tensor2.numel()}"
    )
    print(tensor2)


def recursive_compare(out1, out2, atol, rtol, top_level=True):
    'recursive compare two output'
    if top_level:
        if not isinstance(out1, (list, tuple)):
            return recursive_compare([out1], [out2], atol, rtol, top_level=True)
        all_results = []
        for i, (value1, value2) in enumerate(zip(out1, out2)):
            result = recursive_compare(
                value1, value2, atol, rtol, top_level=False
            )
            all_results.append(result)
            if not result:
                print(f"........... output {i} is not close ........")
                if isinstance(value1, torch.Tensor):
                    print_tensors_diff(value1, value2, atol, rtol)
                else:
                    print(f"{value1} vs {value2}")
        if not all(all_results):
            print(f"all_outputs_compare_result={all_results}")
            return False
        return True

    if isinstance(out1, (list, tuple)):
        for i, (value1, value2) in enumerate(zip(out1, out2)):
            if not recursive_compare(value1, value2, atol, rtol, top_level=False):
                return False
        return True
    if isinstance(out1, torch.Tensor):
        return torch.allclose(
            out1.to(out2.device).to(out2.dtype),
            out2,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )
    return out1 == out2


def recursive_print(args, top_level=True):
    'recursive print args, top_level=True will print index then go to new line'
    def format_tensor(tensor):
        nan_num, inf_num = (
            torch.isnan(tensor).sum().item(),
            torch.isinf(tensor).sum().item(),
        )
        head = "[WARNING]" if (nan_num or inf_num) else ""
        warnings = (
            f" nan_num={nan_num}, inf_num={inf_num}" if (nan_num or inf_num) else ""
        )
        out_str = f"{head} Tensor <shape={tensor.shape}, stride={tensor.stride()},\
             dtype={tensor.dtype}, device={tensor.device}, size={tensor.numel()},{warnings}>"
        return out_str

    out_str = ""
    if isinstance(args, torch.Tensor):
        out_str += format_tensor(args)
    elif isinstance(args, (list, tuple)):
        for i, x in enumerate(args):
            out_str += (
                f"{i}: {recursive_print(x, False)}, \n"
                if top_level
                else f"{recursive_print(x, False)}, "
            )
        if not top_level:
            out_str = f"[{out_str}]" if isinstance(args, list) else f"({out_str})"
    else:
        out_str += str(args)

    if top_level:
        print(out_str)

    return out_str


def compare_single_op(args_path, op_func, atol, rtol):
    'compare single op with saved data'
    with open(args_path, "rb") as f:
        args_cpu = pickle.load(f)
    try:
        output_cpu = op_func(*args_cpu)
    except Exception as excp:
        print(excp)
        print("Convert to float32 ...")
        args_cpu_fp32 = convert_to_dtype(args_cpu, torch.float32)
        output_cpu = op_func(*args_cpu_fp32)

    args_musa = convert_to_musa(args_cpu)
    output_musa = op_func(*args_musa)

    print("....... input .........")
    recursive_print(args_musa)
    print("...... output ........")
    recursive_print(output_musa)

    print("\n...... compare with cpu .......")
    correct = recursive_compare(output_musa, output_cpu, atol, rtol)
    if correct:
        print(
            f"{op_func.__module__}...{op_func.__name__} succeeds to pass CompareWithCPU test"
        )
    else:
        print(
            f"[ERROR] {op_func.__module__}...{op_func.__name__} fails to pass CompareWithCPU test"
        )


class CompareWithCPU(TorchDispatchMode):
    'CompareWithCPU'
    def __init__(
        self,
        enabled=True,
        atol=0.001,
        rtol=0.001,
        target_list=None,
        white_list=None,
        dump_error_data=False,
        verbose=False,
    ) -> None:
        """
        enabled: bool, enable or not
        atol: absolute tolerance
        rtol: relative tolerance
        target_list: if target_list is not empty, then only compare ops in target_list.
        white_list: if target_list is empty, then ops in white_list will be ignored.
        dump_error_data: 
            bool, if dump_error_data is True, all args of the first op \
                which is not all_close with cpu will be saved and then exit
        verbose: 
            bool, if verbose is True, details (tensor shape, dtype, strides) of args will be printed
        Usage:
            with CompareWithCPU():
                run_your_code_here()
        """
        super().__init__()
        self.enabled = enabled
        if not self.enabled:
            return
        self.atol = atol
        self.rtol = rtol
        self.white_list = [
            "_record_function_enter_new.default",
            "detach.default",
            "empty.memory_format",
            "uniform_.default",
            "set_.source_Storage",
            "set_.source_Storage_storage_offset",
            "new_empty.default",
            "random_.default",
            "isinf.default",
            "isnan.default",
        ]
        if white_list is None:
            white_list = []
        self.white_list += white_list
        if target_list is None:
            target_list = []
        self.target_list = target_list
        self.dump_error_data = dump_error_data
        self.verbose = verbose

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print("\n\n============================")
        print(f"{func.__module__}...{func.__name__}")
        if len(self.target_list) > 0 and func.__name__ not in self.target_list:
            print("not in target list, pass")
            return func(*args, **kwargs)
        if func.__name__ in self.white_list:
            print("in white list, pass")
            return func(*args, **kwargs)
        args_cpu = convert_to_cpu(args)
        kwargs_cpu = convert_to_cpu(kwargs)
        out = func(*args, **kwargs)
        if self.verbose:
            print("....... input .........")
            recursive_print(args)
            print("...... output ........")
            recursive_print(out)

        print("\n...... compare with cpu .......")
        correct = False
        try:
            try:
                out_cpu = func(*args_cpu, **kwargs_cpu)
            except Exception as excp:
                print(excp)
                print("Convert to float32 ...")
                args_cpu = convert_to_dtype(args_cpu, torch.float32)
                out_cpu = func(*args_cpu, **kwargs_cpu)
            correct = recursive_compare(out, out_cpu, self.atol, self.rtol)
            if correct:
                print(
                    f"{func.__module__}...{func.__name__} succeeds to pass CompareWithCPU test"
                )
            else:
                print(
                    f"[ERROR] {func.__module__}...{func.__name__} fails to pass CompareWithCPU test"
                )
        except Exception as excp:
            print(excp)

        if self.dump_error_data and not correct:
            with open(f"{func.__name__}_args.pkl", "wb") as f:
                pickle.dump(convert_to_cpu(args), f)
            with open(f"{func.__name__}_kwargs.pkl", "wb") as f:
                pickle.dump(convert_to_cpu(kwargs), f)
            with open(f"{func.__name__}_out.pkl", "wb") as f:
                pickle.dump(convert_to_cpu(out), f)
            print(
                f"input data saved to {func.__name__}_args.pkl,\
                     {func.__name__}_kwargs.pkl, {func.__name__}_out.pkl"
            )
            raise Exception('CompareWithCPU Failed!')

        return out

    def __enter__(self):
        if not self.enabled:
            return None
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return None
        return super().__exit__(exc_type, exc_val, exc_tb)
