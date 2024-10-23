from os.path import dirname, abspath, join
import sys
from typing import List, Set
import yaml

from torchgen.gen import LineLoader
from torchgen.model import DispatchKey

sys.path.append(dirname(dirname(abspath(__file__))))
from codegen.utils import (
    get_pytorch_source_directory,
    get_torch_musa_source_directory,
    flatten_dispatch,
)


def this_dir() -> str:
    return dirname(abspath(__file__))


def parse_dispatch(raw_dispatch) -> dict:
    if raw_dispatch is None:
        return {}
    new_dispatch = flatten_dispatch(raw_dispatch)
    return {DispatchKey.parse(k): v for k, v in new_dispatch.items()}


def get_torch_ops() -> List[dict]:
    yaml_file = join(
        get_pytorch_source_directory(),
        "aten/src/ATen/native/native_functions.yaml",
    )
    torch_ops: List[dict] = []
    with open(yaml_file, "r", encoding="utf-8") as f:
        _ops = yaml.load(f, Loader=LineLoader)
        for op in _ops:
            op.pop("__line__", None)
            dispatch = op.pop("dispatch", None)
            op["dispatch"] = parse_dispatch(dispatch)
            op_name = op["func"].split("(")[0].strip()
            op["func"] = op_name
            torch_ops.append(op)
    return torch_ops


def get_musa_ops() -> Set[str]:
    yaml_file = join(
        get_torch_musa_source_directory(),
        "torch_musa/csrc/aten/ops/musa_functions.yaml",
    )
    musa_ops: Set[str] = set()
    with open(yaml_file, "r", encoding="utf-8") as f:
        _ops = yaml.load(f, Loader=LineLoader)
        for op in _ops:
            op_name = op["func"].strip()
            musa_ops.add(op_name)
    return musa_ops
