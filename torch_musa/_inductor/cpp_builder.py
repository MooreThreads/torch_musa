"""MUSA CPP Builder"""

import os
from typing import List, Sequence, Tuple
import torch
from torch._inductor.cpp_builder import (
    CppTorchOptions,
    _append_list,
)
from torch._inductor.cpu_vec_isa import invalid_vec_isa, VecISA
from torch_musa.utils import musa_extension


def get_cpp_torch_musa_options(
    musa: bool,
    aot_mode: bool = False,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Retrieve the C++ compilation and linking options needed for building the Torch-MUSA extension.
    :param musa (bool):
        Indicates whether to enable the MUSA backend.
    :param aot_mode (bool, default False):
        If True, apply additional path transformations for AOT mode.
    Returns:
        A tuple of seven lists in the following order:
            1. definations: List of preprocessing macro definitions.
            2. include_dirs: List of include directory paths.
            3. cflags: List of C/C++ compiler flags.
            4. ldflags: List of linker flags.
            5. libraries_dirs: List of library search paths.
            6. libraries: List of libraries to link against.
            7. passthough_args: List of extra arguments passed directly to the underlying compiler.
    """
    definations: List[str] = []
    include_dirs: List[str] = []
    cflags: List[str] = []
    ldflags: List[str] = []
    libraries_dirs: List[str] = []
    libraries: List[str] = []
    passthough_args: List[str] = []

    include_dirs = musa_extension.include_paths(musa)
    libraries_dirs = musa_extension.library_paths(musa)

    if musa:
        definations.append(" USE_ROCM" if torch.version.hip else " USE_MUSA")

        # libraries += ["c10_musa", "musa", "torch_musa"]
        libraries += ["musa", "musa_python"]

    if aot_mode:

        if musa and torch.version.hip is None:
            _transform_musa_paths(libraries_dirs)

    return (
        definations,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthough_args,
    )


def _transform_musa_paths(lpaths: List[str]) -> None:
    # This handles two cases:
    # 1. Meta internal cuda-12 where libs are in lib/cuda-12 and lib/cuda-12/stubs
    # 2. Linux machines may have CUDA installed under either lib64/ or lib/
    for i, path in enumerate(lpaths):
        if (
            "MUSA_HOME" in os.environ
            and path.startswith(os.environ["MUSA_HOME"])
            and not os.path.exists(f"{path}/libmusart_static.a")
        ):
            for root, _, files in os.walk(path):
                if "libmusart_static.a" in files:
                    lpaths[i] = os.path.join(path, root)
                    lpaths.append(os.path.join(lpaths[i], "stubs"))
                    break


class CppTorchMusaOptions(CppTorchOptions):
    """
    This class is inherited from CppTorchOptions, which automatic contains
    base cxx build options and torch common build options. And then it will
    maintains cuda device related build args.
    """

    def __init__(
        self,
        vec_isa: VecISA = invalid_vec_isa,
        include_pytorch: bool = False,
        musa: bool = True,
        aot_mode: bool = False,
        compile_only: bool = False,
        use_absolute_path: bool = False,
        use_mmap_weights: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
    ) -> None:
        super().__init__(
            vec_isa=vec_isa,
            include_pytorch=include_pytorch,
            aot_mode=aot_mode,
            compile_only=compile_only,
            use_absolute_path=use_absolute_path,
            use_mmap_weights=use_mmap_weights,
            extra_flags=extra_flags,
        )

        musa_definations: List[str] = []
        musa_include_dirs: List[str] = []
        musa_cflags: List[str] = []
        musa_ldflags: List[str] = []
        musa_libraries_dirs: List[str] = []
        musa_libraries: List[str] = []
        musa_passthough_args: List[str] = []

        (
            musa_definations,
            musa_include_dirs,
            musa_cflags,
            musa_ldflags,
            musa_libraries_dirs,
            musa_libraries,
            musa_passthough_args,
        ) = get_cpp_torch_musa_options(musa=musa, aot_mode=aot_mode)
        _append_list(self._definations, musa_definations)
        _append_list(self._include_dirs, musa_include_dirs)
        _append_list(self._cflags, musa_cflags)
        _append_list(self._ldflags, musa_ldflags)
        _append_list(self._libraries_dirs, musa_libraries_dirs)
        _append_list(self._libraries, musa_libraries)
        _append_list(self._passthough_args, musa_passthough_args)
        self._finalize_options()
