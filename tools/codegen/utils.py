"""Utilities classes/functions for musa codegen"""

from copy import deepcopy
from os import getenv, listdir
from os.path import isdir, join, dirname, abspath, exists
from shutil import copy
from typing import Set, Dict

from torchgen.model import DispatchKey

from codegen.model import MUSA_STRUCTURED_DISPATCH_KEY


def flatten_dispatch(dispatch: Dict[str, str]) -> Dict[str, str]:
    """Separate multiple dispatch keys that share the same kernel"""
    assert dispatch is not None
    assert len(dispatch) > 0
    flatten: Dict[str, str] = {}

    for dk_str, kernel in dispatch.items():
        if dk_str == "__line__":
            continue
        for k in dk_str.split(","):
            k = k.strip()
            assert k != "" and k not in flatten
            flatten[k] = deepcopy(kernel)
    return flatten


def get_pytorch_source_directory() -> str:
    """Get pytorch source directory from env variables"""
    pytorch_source = getenv("PYTORCH_PATH", getenv("PYTORCH_REPO_PATH", ""))
    if not isdir(pytorch_source):
        pytorch_source = join(getenv("TORCH_MUSA_HOME", ""), "../pytorch")
    assert isdir(pytorch_source), "pytorch source directory doesn't exist."
    return pytorch_source


def get_pytorch_install_directory() -> str:
    """Get pytorch installation directory"""
    install_directory: str
    try:
        import torch  # pylint: disable=C0415

        install_directory = dirname(torch.__file__)
    except Exception as exc:
        raise ImportError("pytorch installation not found.") from exc
    return install_directory


def get_torch_musa_source_directory() -> str:
    """Get torch_musa source directory"""
    torch_musa_source = getenv("TORCH_MUSA_HOME", "")
    if not isdir(torch_musa_source):
        torch_musa_source = abspath(join(__file__, "..", "..", ".."))
    assert isdir(torch_musa_source), "torch_musa source directory doesn't exist."
    return torch_musa_source


class TorchOpsHeadersMigrator:
    """
    Kernels listed in `musa_functions.yaml` have the following characteristics:
      1. May be implemented by adopting a different approach than pytorch
         (e.g. torch-structured but musa-unstructured).
      2. Default kernel namespace is `at::musa`, which is different from pytorch (`at::native`).
      3. In some cases, it may be possible to directly invoke the kernel for another special
         backend (e.g. CPU) for avoiding the overhead for dispatch.
    To accommodate the differences between pytorch and torch_musa for compatibility purposes, this
    class classifies all existing pytorch kernels header files and migrates them to the directory
    tree of torch_musa building:
      1. *.h/*_ops.h: backend independent files, just copy.
      2. *_native.h/*_meta.h: backend dependent files. To maintain the consistency in header files
         inclusion, rename as `{wrap_prefix}*_[native|meta].h` and copy, create `*_[native|meta].h`
         and include these `wrapped` old header files, finally add musa's function signatures.
      3. *_{backend}_dispatch.h: for common functions keys (e.g. CPU、CompositeImplicitAutograd),
         just copy these backend-related header files for the purpose of direct invocation.
      4. Other files (e.g. `from_blob.h` and `tensor.h`), just `mechanical` selete and copy.
    """

    def __init__(
        self,
        *,
        wrap_prefix: str,
        functions_keys: Set[DispatchKey],
    ) -> None:
        assert isinstance(wrap_prefix, str) and wrap_prefix != ""
        self.wrap_prefix = wrap_prefix

        assert DispatchKey.CUDA not in functions_keys
        self.functions_keys = deepcopy(functions_keys)
        self.functions_keys.remove(MUSA_STRUCTURED_DISPATCH_KEY)

        self.torch_ops_install_dir: str = join(
            get_pytorch_install_directory(),
            "include",
            "ATen",
            "ops",
        )
        self.musa_ops_gen_dir: str = join(
            get_torch_musa_source_directory(),
            "build",
            "torch_musa_codegen",
            "ATen",
            "ops",
        )
        self.extra_headers = [
            "from_blob.h",
            "tensor.h",
        ]

        self.root_names: Set[str] = self.get_torch_all_ops_root_names()
        self.musa_meta_names: Set[str] = set()
        self.musa_native_names: Set[str] = set()

    def get_torch_all_ops_root_names(self) -> Set[str]:
        """Collect all torch kernels root names listed in `native_functions.yaml`"""
        torch_ops_files = listdir(self.torch_ops_install_dir)
        ops_root_names: Set[str] = set()

        for op_file in torch_ops_files:
            if not op_file.endswith("_ops.h"):
                continue
            root_name = op_file.replace("_ops.h", "")
            assert root_name not in ops_root_names
            ops_root_names.add(root_name)

        return ops_root_names

    def write_if_changed(
        self,
        new_path: str,
        old_path: str,
    ) -> None:
        """Files already exist, override if contents are different."""
        old_contents: str
        new_contents: str
        with open(old_path, "r", encoding="utf-8") as f:
            old_contents = f.read()
        with open(new_path, "r", encoding="utf-8") as f:
            new_contents = f.read()
        if old_contents != new_contents:
            copy(new_path, old_path)

    def copy_file(
        self,
        torch_file: str,
        musa_file: str,
    ) -> None:
        """
        Loose restriction:
          1. `musa_file` does not exist, do copy.
          2. Different contents between two files, do replace.
        """
        torch_path = join(self.torch_ops_install_dir, torch_file)
        if not exists(torch_path):
            return
        musa_path = join(self.musa_ops_gen_dir, musa_file)
        if not exists(musa_path):
            copy(torch_path, musa_path)
        else:
            self.write_if_changed(torch_path, musa_path)

    def wrap_file_name(self, file_name: str) -> str:
        return self.wrap_prefix + file_name

    def add_musa_meta_root_name(self, root_name: str) -> None:
        assert root_name not in self.musa_meta_names
        self.musa_meta_names.add(root_name)

    def add_musa_native_root_name(self, root_name: str) -> None:
        assert root_name not in self.musa_native_names
        self.musa_native_names.add(root_name)

    def migrate_op_h(self, root_name: str) -> None:
        file_name = f"{root_name}.h"
        self.copy_file(file_name, file_name)

    def migrate_op_ops_h(self, root_name: str) -> None:
        file_name = f"{root_name}_ops.h"
        self.copy_file(file_name, file_name)

    def migrate_op_meta_h(self, root_name: str) -> None:
        torch_file = f"{root_name}_meta.h"
        musa_file: str
        if root_name in self.musa_meta_names:
            musa_file = self.wrap_file_name(torch_file)
        else:
            musa_file = torch_file
        self.copy_file(torch_file, musa_file)

    def migrate_op_native_h(self, root_name: str) -> None:
        torch_file = f"{root_name}_native.h"
        musa_file: str
        if root_name in self.musa_native_names:
            musa_file = self.wrap_file_name(torch_file)
        else:
            musa_file = torch_file
        self.copy_file(torch_file, musa_file)

    def migrate_op_dispatch_h(self, root_name: str) -> None:
        for func_key in self.functions_keys:
            file_name = f"{root_name}_{func_key.lower()}_dispatch.h"
            self.copy_file(file_name, file_name)

    def finish_extra_migration(self) -> None:
        for file in self.extra_headers:
            self.copy_file(file, file)

    def migrate_op_headers(self) -> None:
        for root_name in self.root_names:
            self.migrate_op_meta_h(root_name)
            self.migrate_op_native_h(root_name)

            self.migrate_op_h(root_name)
            self.migrate_op_ops_h(root_name)
            self.migrate_op_dispatch_h(root_name)
        self.finish_extra_migration()
