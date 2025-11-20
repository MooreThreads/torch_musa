"""AOTI Package"""

import os
import tempfile
from typing import Callable

import torch
import torch._inductor.package
import torch.utils._pytree as pytree
from torch._inductor.package.package import PT2ArchiveReader, compile_so
from torch._inductor.package.pt2_archive_constants import AOTINDUCTOR_DIR
from torch.export._tree_utils import reorder_kwargs

import torch_musa

# pylint: disable=no-else-raise


def load_package(path: str, device: str) -> Callable:  # type: ignore[type-arg]
    """
    Load an AOT Inductor–compiled model package and return an optimized runner function.

    Parameters
    ----------
    path : str
        Path to the model package. Must be either:
        - A .pt2 archive file (AOT Inductor package).
        - A directory containing extracted AOT Inductor files.
    device : str
        Target device for execution. Supported values:
        - "cpu" for CPU execution.
        - "musa" or any string starting with "musa:" for MUSA devices.

    Returns
    -------
    Callable
        A function that wraps the AOTI runner. Its signature matches the original model:
        ``outputs = optimized(*args, **kwargs)``.

    """

    def _get_aoti_file_with_suffix(suffix: str) -> str:
        for file in path:
            if file.endswith(suffix):
                return file
        raise RuntimeError(f"Unable to find file with suffix {suffix}")

    if isinstance(path, list):
        so_path = _get_aoti_file_with_suffix(".so")
    else:
        if path.endswith(".so"):
            raise RuntimeError(
                "Unable to load .so. It should be a .pt2 format or a directory."
            )

        elif path.endswith(".pt2"):
            so_path = os.path.splitext(path)[0]
            with PT2ArchiveReader(path) as archive_reader:
                file_names = archive_reader.get_file_names()

                with tempfile.TemporaryDirectory() as tmp_dir:
                    archive_reader.extractall(tmp_dir)
                    file_names = archive_reader.get_file_names()
                    aoti_files = [
                        file for file in file_names if file.startswith(AOTINDUCTOR_DIR)
                    ]

                    so_path = compile_so(tmp_dir, aoti_files, so_path)

        else:
            assert os.path.isdir(path), "Must specify a directory or a .pt2 file"
            aoti_files = [
                os.path.join(root, file)
                for root, dirs, files in os.walk(path)
                for file in files
            ]
            so_path = compile_so(path, aoti_files, path)

    if device == "cpu":
        runner = torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)  # type: ignore[call-arg]
    elif device == "musa" or device.startswith("musa:"):
        runner = torch_musa._MUSAC._aoti.AOTIModelContainerRunnerMusa(
            so_path, 1, device
        )
    else:
        raise RuntimeError("Unsupported device " + device)

    def optimized(*args, **kwargs):  # type: ignore[no-untyped-def]
        call_spec = runner.get_call_spec()  # type: ignore[attr-defined]
        in_spec = pytree.treespec_loads(call_spec[0])
        out_spec = pytree.treespec_loads(call_spec[1])
        flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
        flat_outputs = runner.run(flat_inputs)  # type: ignore[attr-defined]
        return pytree.tree_unflatten(flat_outputs, out_spec)

    return optimized


def _apply_package_patch():
    torch._inductor.package.package.load_package = load_package


_apply_package_patch()
