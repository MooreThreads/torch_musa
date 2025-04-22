from typing import List

def is_deprecated(torch_op: dict) -> bool:
    op_name = torch_op["func"]
    return op_name.startswith("_cast_") or \
        op_name == "set_data" or \
        op_name.startswith("_aminmax.dim") or \
        op_name.startswith("nuclear_norm")

def is_cudnn_like(torch_op: dict) -> bool:
    op_name = torch_op["func"]
    return "cudnn" in op_name.lower()

def is_only_for_testing(torch_op: dict) -> bool:
    op_name = torch_op["func"]
    return op_name.startswith("_test_")

def is_other_conv(torch_op: dict) -> bool:
    op_name = torch_op["func"]
    if op_name in ["convolution_overrideable", "convolution_backward_overrideable"]:
        return False
    return op_name.startswith("convolution") or \
        op_name.startswith("_convolution") or \
        op_name.startswith("conv1d") or \
        op_name.startswith("conv2d") or \
        op_name.startswith("conv3d") or \
        op_name.startswith("conv_tbc") or \
        op_name.startswith("conv_transpose") or \
        op_name.startswith("slow_conv") or \
        op_name.startswith("_slow_conv")

def is_other_attention(torch_op: dict) -> bool:
    op_name = torch_op["func"]
    return op_name == "_fused_sdp_choice" or \
        "_attention" in op_name

def is_complex_like(torch_op: dict) -> bool:
    op_name = torch_op["func"]
    return op_name.startswith("complex")

def is_in_special_module(torch_op: dict) -> bool:
    op_name = torch_op["func"]
    return op_name.startswith("special_")

RULES = [
    is_deprecated,
    is_cudnn_like,
    is_only_for_testing,
    is_other_conv,
    is_other_attention,
    is_complex_like,
    is_in_special_module,
]

def clean_torch_ops(torch_ops_list: List[dict]) -> List[dict]:
    is_negligible = lambda op: any([r(op) for r in RULES])

    clean_ops = [op for op in torch_ops_list if not is_negligible(op)]
    return clean_ops
