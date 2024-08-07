"""Utilities for integration module"""

from enum import Enum
from typing import List, NoReturn, Dict, Optional
import tarfile
import os
import random
import re

import numpy as np

import torch
import torch_musa


class ExtendedEnum(Enum):
    @classmethod
    def choices(cls) -> List[str]:
        return [str(v) for v in cls]

    @classmethod
    def text_map(cls) -> Dict[str, "ExtendedEnum"]:
        return {str(v): v for v in cls}


def assert_never(x: NoReturn) -> NoReturn:
    raise AssertionError(f"Unhandled type: {type(x).__name__}")


def decompress(path: str) -> None:
    target_dir: str = os.path.abspath(os.path.dirname(path))
    if path.endswith("tar.gz"):
        with tarfile.open(path, "r:gz") as f:
            f.extractall(path=target_dir)


_NUM_DEVICES = torch_musa.device_count()
_DEVICE_PATTERN = re.compile(r"^musa(?::(0|[1-9]\d*))?$")


def check_device(device: str) -> None:
    res = _DEVICE_PATTERN.match(device)
    flag = False
    if res:
        idx = res.group(1)
        if (idx is None) or (int(idx) < _NUM_DEVICES):
            flag = True
    if not flag:
        raise ValueError(f"Invalid device `{device}`")


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.musa.manual_seed(seed)
    torch.musa.manual_seed_all(seed)


class Reference:
    def __init__(self, value) -> None:
        self.container = [value]

    def update(self, func) -> None:
        func(self.container[0])

    def current(self):
        return self.container[0]


def check_existent_directory(
    dir_path: str,
    error_message: Optional[str] = None,
) -> None:
    if not os.path.isdir(dir_path):
        if (error_message is None) or (error_message == ""):
            error_message = f"Not found directory: `{dir_path}`"
        raise FileNotFoundError(error_message)


def check_existent_file(
    file_path: str,
    error_message: Optional[str] = None,
) -> None:
    if not os.path.isfile(file_path):
        if (error_message is None) or (error_message == ""):
            error_message = f"Not found file: `{file_path}`"
        raise FileNotFoundError(error_message)


_DATA_ROOT_ENV_KEY = "INTEGRATION_DATA_ROOT"


def get_data_root_dir() -> str:
    if _DATA_ROOT_ENV_KEY not in os.environ:
        raise RuntimeError(
            f"Not found data root env key `{_DATA_ROOT_ENV_KEY}` for integration"
        )
    root_dir = os.environ.get(_DATA_ROOT_ENV_KEY)
    check_existent_directory(
        root_dir, f"Not found integration data root dir: `{root_dir}`"
    )
    return root_dir


def get_dataset_root_dir() -> str:
    root_dir = os.path.join(get_data_root_dir(), "datasets")
    check_existent_directory(
        root_dir, f"Not found integration dataset root dir: `{root_dir}`"
    )
    return root_dir


def get_model_root_dir() -> str:
    root_dir = os.path.join(get_data_root_dir(), "models")
    check_existent_directory(
        root_dir, f"Not found integration model root dir: `{root_dir}`"
    )
    return root_dir
