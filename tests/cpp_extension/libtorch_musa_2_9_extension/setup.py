import distutils.command.clean
import shutil
import sys
from pathlib import Path

from setuptools import find_packages, setup

from torch.utils.cpp_extension import IS_WINDOWS
from torch_musa.utils.musa_extension import BuildExtension, MUSAExtension


ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR.parent))

from libtorch_musa_porting import (  # noqa: E402
    DEFAULT_CSRC_SUFFIXES,
    DEFAULT_PACKAGE_SUFFIXES,
    PYTORCH_DIR,
    port_tree,
)

PACKAGE_DIR = ROOT_DIR / "libtorch_agn_2_9"
GENERATED_CSRC_DIR = ROOT_DIR / "build" / "generated_csrc"

PYTORCH_2_9_CSRC_DIR = PYTORCH_DIR / "test/cpp_extensions/libtorch_agn_2_9_extension/csrc"
PYTORCH_2_9_PACKAGE_DIR = PYTORCH_DIR / "test/cpp_extensions/libtorch_agn_2_9_extension/libtorch_agn_2_9"

CSRC_PORTING_REPLACEMENTS = (
    ("__CUDACC__", "__MUSACC__"),
    ("#include <cuda_runtime.h>", "#include <musa_runtime.h>"),
    ("LAE_USE_CUDA", "LAE_USE_MUSA"),
    (
        "STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CUDA, m)",
        "STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, PrivateUse1, m)",
    ),
    ("cudaError_t", "musaError_t"),
    ("cudaGetDevice", "musaGetDevice"),
    ("cudaSuccess", "musaSuccess"),
    ("Accessor_cuda", "Accessor_musa"),
    ("mv_tensor_accessor_cuda", "mv_tensor_accessor_musa"),
)


def prepare_ported_package():
    port_tree(
        PYTORCH_2_9_PACKAGE_DIR,
        PACKAGE_DIR,
        allowed_suffixes=DEFAULT_PACKAGE_SUFFIXES,
        source_name="package",
    )


def prepare_ported_csrc():
    return port_tree(
        PYTORCH_2_9_CSRC_DIR,
        GENERATED_CSRC_DIR,
        allowed_suffixes=DEFAULT_CSRC_SUFFIXES,
        source_name="csrc",
        text_replacements=CSRC_PORTING_REPLACEMENTS,
    )


prepare_ported_package()


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove extension
        for path in (ROOT_DIR / "libtorch_agn_2_9").glob("**/*.so"):
            path.unlink()
        # Remove build and dist and egg-info directories
        dirs = [
            ROOT_DIR / "build",
            ROOT_DIR / "dist",
            ROOT_DIR / "libtorch_agn_2_9.egg-info",
        ]
        for path in dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def get_extension():
    source_dir = prepare_ported_csrc()
    extra_compile_args = {
        "cxx": [
            "-DTORCH_STABLE_ONLY",
            "-DTORCH_TARGET_VERSION=0x0209000000000000",
            "-DSTABLE_LIB_NAME=libtorch_agn_2_9",
            "-DLAE_USE_MUSA",
        ],
        "mcc": [
            "-O2",
            "-DTORCH_TARGET_VERSION=0x0209000000000000",
            "-DSTABLE_LIB_NAME=libtorch_agn_2_9",
        ],
    }
    if not IS_WINDOWS:
        extra_compile_args["cxx"].append("-fdiagnostics-color=always")

    sources = list(source_dir.glob("**/*.cpp"))
    sources.extend(source_dir.glob("**/*.mu"))

    return [
        MUSAExtension(
            "libtorch_agn_2_9._C",
            sources=sorted(str(s) for s in sources),
            py_limited_api=True,
            extra_compile_args=extra_compile_args,
            extra_link_args=[],
            # libraries=["musart", "musa"],
        )
    ]


setup(
    name="libtorch_agn_2_9",
    version="0.0",
    author="MUSA",
    description="Example of libtorch agnostic extension for PyTorch 2.9",
    packages=find_packages(exclude=("test",)),
    package_data={"libtorch_agn_2_9": ["*.dll", "*.dylib", "*.so"]},
    install_requires=[
        "torch",
    ],
    ext_modules=get_extension(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        "clean": clean,
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
